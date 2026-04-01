/**
 * chronos_dual_gpu_daemon.cpp
 *
 * DUAL-GPU PRODUCER-CONSUMER PIPELINE
 * =====================================
 * GPU 0 (780M)  : PRODUCER  — loads + normalizes packet data, builds staging buffers
 * GPU 1 (7700S) : CONSUMER  — runs Bi-Twistor observer HLSL kernel, writes results
 *
 * While the 7700S processes dataset[N], the 780M is already staging dataset[N+1].
 * Neither chip sits idle between jobs.
 *
 * BYPASS STACK:
 *   AmdPowerXpressRequestHighPerformance = 1   → AMD driver: prefer discrete GPU
 *   DX11 SwapChain on 7700S device             → OS "3D program" signal, keeps 7700S awake
 *   780M gets its own DX11 device (no swap)    → always awake as system default
 *   SelectAdapter() loops EnumAdapters by VRAM → never assumes adapter index
 *
 * Python interface: stdin JSON lines
 *   {"packets": "path/to/merged.json", "out": "path/to/results_dir"}
 *   blank line → shutdown
 *   Responses on stdout: "READY", "DONE:<dir>", "ERROR:<msg>"
 *
 * Build:
 *   cl /O2 /EHsc /std:c++17 chronos_dual_gpu_daemon.cpp /Fe:chronos_dual_gpu_daemon.exe
 *      d3d11.lib dxgi.lib d3dcompiler.lib user32.lib gdi32.lib ole32.lib
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "user32.lib")

// ── HARDWARE BYPASS ───────────────────────────────────────────────────────────
extern "C" __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

using namespace std;
namespace fs = std::filesystem;

const UINT AMD_VENDOR_ID     = 0x1002;
const UINT RX7700S_DEVICE_ID = 0x7480;  // discrete

// ── Staged job: output from 780M producer, input to 7700S consumer ───────────
struct StagedJob {
    string              label;
    string              out_dir;
    UINT                num_bins;
    vector<float>       streams;   // [bins * 4 streams * 4 features]
    vector<uint32_t>    mask;      // [bins * 4]
};

// ── Thread-safe job queue ─────────────────────────────────────────────────────
struct JobQueue {
    queue<StagedJob>     q;
    mutex                mtx;
    condition_variable   cv;
    atomic<bool>         done{false};

    void push(StagedJob j) {
        { lock_guard<mutex> lk(mtx); q.push(move(j)); }
        cv.notify_one();
    }
    bool pop(StagedJob& out) {
        unique_lock<mutex> lk(mtx);
        cv.wait(lk, [&]{ return !q.empty() || done; });
        if (q.empty()) return false;
        out = move(q.front()); q.pop(); return true;
    }
    void finish() { done = true; cv.notify_all(); }
};

// ── Global state ──────────────────────────────────────────────────────────────
struct GPUDevice {
    ID3D11Device*        dev  = nullptr;
    ID3D11DeviceContext* ctx  = nullptr;
    IDXGISwapChain*      swap = nullptr; // only for 7700S
    IDXGIAdapter*        adp  = nullptr;
    string               name;
};

GPUDevice g_780m, g_7700s;
IDXGIFactory* g_factory = nullptr;
ID3D11ComputeShader* g_cs = nullptr;

HWND      g_hwnd  = nullptr;
HINSTANCE g_hInst = nullptr;

// ── Win32 dummy ───────────────────────────────────────────────────────────────
LRESULT CALLBACK DummyWndProc(HWND h, UINT m, WPARAM w, LPARAM l) { return DefWindowProc(h,m,w,l); }

// ── JSON field extractors (no external lib) ───────────────────────────────────
double JsonDouble(const string& s, const string& key, double def=0.0) {
    auto p=s.find('"'+key+'"'); if(p==string::npos) return def;
    auto c=s.find(':',p);       if(c==string::npos) return def;
    try { return stod(s.substr(c+1)); } catch(...) { return def; }
}
string JsonStr(const string& s, const string& key) {
    auto p=s.find('"'+key+'"'); if(p==string::npos) return "";
    auto q=s.find('"',p+key.size()+3); if(q==string::npos) return "";
    auto r=s.find('"',q+1); if(r==string::npos) return "";
    return s.substr(q+1,r-q-1);
}

// ── Buffer helpers (device-agnostic) ─────────────────────────────────────────
static ID3D11Buffer* MakeBuf(ID3D11Device* dev, UINT stride, UINT count,
                               const void* data=nullptr, bool uav=false) {
    D3D11_BUFFER_DESC bd{};
    bd.ByteWidth=stride*count; bd.StructureByteStride=stride;
    bd.Usage=D3D11_USAGE_DEFAULT; bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.BindFlags=uav?(D3D11_BIND_UNORDERED_ACCESS|D3D11_BIND_SHADER_RESOURCE):D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd{data}; ID3D11Buffer* b=nullptr;
    dev->CreateBuffer(&bd,data?&sd:nullptr,&b); return b;
}
static ID3D11ShaderResourceView* MakeSRV(ID3D11Device* dev, ID3D11Buffer* buf) {
    D3D11_SHADER_RESOURCE_VIEW_DESC sd{}; sd.ViewDimension=D3D11_SRV_DIMENSION_BUFFEREX;
    D3D11_BUFFER_DESC bd; buf->GetDesc(&bd); sd.BufferEx.NumElements=bd.ByteWidth/bd.StructureByteStride;
    ID3D11ShaderResourceView* s=nullptr; dev->CreateShaderResourceView(buf,&sd,&s); return s;
}
static ID3D11UnorderedAccessView* MakeUAV(ID3D11Device* dev, ID3D11Buffer* buf) {
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;
    D3D11_BUFFER_DESC bd; buf->GetDesc(&bd); ud.Buffer.NumElements=bd.ByteWidth/bd.StructureByteStride;
    ID3D11UnorderedAccessView* u=nullptr; dev->CreateUnorderedAccessView(buf,&ud,&u); return u;
}
static vector<float> Readback(ID3D11Device* dev, ID3D11DeviceContext* ctx,
                               ID3D11Buffer* src, UINT count) {
    D3D11_BUFFER_DESC bd; src->GetDesc(&bd);
    bd.Usage=D3D11_USAGE_STAGING; bd.BindFlags=0; bd.MiscFlags=0; bd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ID3D11Buffer* st=nullptr; dev->CreateBuffer(&bd,nullptr,&st); ctx->CopyResource(st,src);
    D3D11_MAPPED_SUBRESOURCE ms; ctx->Map(st,0,D3D11_MAP_READ,0,&ms);
    vector<float> out(count); memcpy(out.data(),ms.pData,count*4);
    ctx->Unmap(st,0); st->Release(); return out;
}

// ── 780M PRODUCER: loads + stages a json file ────────────────────────────────
// Returns a StagedJob ready to send to the 7700S queue.
// All CPU/IO work happens here — 7700S is free to compute.
StagedJob Stage(const string& packets_json, const string& out_dir) {
    StagedJob job; job.out_dir = out_dir;
    job.label = fs::path(out_dir).filename().string();

    ifstream jf(packets_json); if(!jf) return job;
    string content((istreambuf_iterator<char>(jf)),{});

    struct Pkt { double t; int wi; float feat[5]; };
    vector<Pkt> pkts;
    size_t pos=0;
    while(true) {
        size_t a=content.find('{',pos); if(a==string::npos) break;
        size_t b=content.find('}',a);  if(b==string::npos) break;
        string obj=content.substr(a,b-a+1);
        double t=JsonDouble(obj,"host_time_mid"); if(t==0.0){pos=b+1;continue;}
        Pkt p; p.t=t; p.wi=max(0,min(3,(int)JsonDouble(obj,"device_worker_index")));
        p.feat[0]=(float)(JsonDouble(obj,"clock_mean")/3e9);
        p.feat[1]=(float)(JsonDouble(obj,"clock_std") /1e6);
        p.feat[2]=(float)(JsonDouble(obj,"clock_count")/1000.0);
        p.feat[3]=(float)p.wi;
        p.feat[4]=(float)(JsonDouble(obj,"bou_h") / 20000.0); // Normalize H field (~20600 nT)
        pkts.push_back(p); pos=b+1;
    }
    if(pkts.empty()) return job;

    // Bin into 100ms slots on 780M (CPU compute)
    const UINT NS=4,FD=5; double t0=pkts[0].t;
    map<int,map<int,Pkt>> bins;
    for(auto& p:pkts){ int bi=(int)((p.t-t0)/0.1); bins[bi][p.wi]=p; }
    job.num_bins=(UINT)bins.size();
    job.streams.assign(job.num_bins*NS*FD, 0.0f);
    job.mask.assign(job.num_bins*NS, 0);
    UINT bi=0;
    for(auto& [b,workers]:bins){
        for(auto& [si,p]:workers){
            job.mask[bi*NS+si]=1;
            for(int f=0;f<5;f++) job.streams[bi*NS*FD+si*FD+f]=p.feat[f];
        }
        ++bi;
    }
    cerr << "[780M] Staged " << pkts.size() << " packets, " << bins.size() << " bins -> " << job.label << endl;
    return job;
}

// ── 7700S CONSUMER: runs the compute kernel ───────────────────────────────────
void Compute(const StagedJob& job, const string& results_dir) {
    CreateDirectoryA(results_dir.c_str(),nullptr);
    const UINT NS=4,FD=5,LATTICE=16,NB=job.num_bins;
    if(NB==0) { cerr << "[7700S] Empty job: " << job.label << endl; return; }

    vector<float> lat(FD*LATTICE,0.f);
    auto pS=MakeBuf(g_7700s.dev,4,NB*NS*FD,job.streams.data());
    auto pM=MakeBuf(g_7700s.dev,4,NB*NS,   job.mask.data());
    auto pSN=MakeBuf(g_7700s.dev,4,NB*FD,  nullptr,true);
    auto pC =MakeBuf(g_7700s.dev,4,NB,     nullptr,true);
    auto pW =MakeBuf(g_7700s.dev,4,NB,     nullptr,true);
    auto pL =MakeBuf(g_7700s.dev,4,FD*LATTICE,lat.data(),true);

    struct CB { uint32_t NB,NS,FD,LAT; float mix,zt; uint32_t sep,pad; };
    CB cb{NB,NS,FD,LATTICE,0.5f,2.0f,10,0};
    D3D11_BUFFER_DESC cbd{sizeof(CB),D3D11_USAGE_IMMUTABLE,D3D11_BIND_CONSTANT_BUFFER};
    D3D11_SUBRESOURCE_DATA cdi{&cb}; ID3D11Buffer* pCB=nullptr;
    g_7700s.dev->CreateBuffer(&cbd,&cdi,&pCB);

    g_7700s.ctx->CSSetShader(g_cs,nullptr,0);
    g_7700s.ctx->CSSetConstantBuffers(0,1,&pCB);
    ID3D11ShaderResourceView* srvs[]={MakeSRV(g_7700s.dev,pS),MakeSRV(g_7700s.dev,pM)};
    g_7700s.ctx->CSSetShaderResources(0,2,srvs);
    ID3D11UnorderedAccessView* uavs[]={MakeUAV(g_7700s.dev,pSN),MakeUAV(g_7700s.dev,pC),
                                       MakeUAV(g_7700s.dev,pW), MakeUAV(g_7700s.dev,pL)};
    g_7700s.ctx->CSSetUnorderedAccessViews(0,4,uavs,nullptr);
    g_7700s.ctx->Dispatch((NB+63)/64,1,1);
    g_7700s.ctx->Flush();

    auto coh=Readback(g_7700s.dev,g_7700s.ctx,pC,NB);
    auto wsc=Readback(g_7700s.dev,g_7700s.ctx,pW,NB);
    float mc=accumulate(coh.begin(),coh.end(),0.f)/(float)NB;
    float mv=accumulate(wsc.begin(),wsc.end(),0.f)/(float)NB;
    float mu=mc,sig=0.f; for(float c:coh) sig+=(c-mu)*(c-mu); sig=sqrt(sig/(float)NB);
    vector<pair<int,float>> peaks;
    for(int i=0;i<(int)coh.size();i++){ float z=sig>0?(coh[i]-mu)/sig:0; if(z>2.f) peaks.push_back({i,z}); }

    cerr << "[7700S] " << job.label << " -> coh=" << mc << " vel=" << mv << " peaks=" << peaks.size() << endl;

    ofstream rf(results_dir+"/results.json");
    rf<<"{\"coherence\":"<<mc<<",\"velocity\":"<<mv<<",\"peak_event_count\":"<<peaks.size()<<",\"peaks\":[";
    for(size_t i=0;i<peaks.size();i++){if(i) rf<<","; rf<<"{\"bin\":"<<peaks[i].first<<",\"z\":"<<peaks[i].second<<"}";}
    rf<<"]}"; rf.close();

    pCB->Release(); for(auto*s:srvs) if(s) s->Release(); for(auto*u:uavs) if(u) u->Release();
    pS->Release(); pM->Release(); pSN->Release(); pC->Release(); pW->Release(); pL->Release();
}

// ── PRODUCER THREAD (780M) ────────────────────────────────────────────────────
void ProducerThread(JobQueue& queue) {
    string line;
    while(getline(cin,line)) {
        if(line.empty()) { queue.finish(); return; }
        string pjson=JsonStr(line,"packets");
        string odir =JsonStr(line,"out");
        if(pjson.empty()||odir.empty()) { cout<<"ERROR:bad job"<<endl; cout.flush(); continue; }
        StagedJob job=Stage(pjson,odir);
        queue.push(move(job));
    }
    queue.finish();
}

// ── CONSUMER THREAD (7700S) ───────────────────────────────────────────────────
void ConsumerThread(JobQueue& queue) {
    StagedJob job;
    while(queue.pop(job)) {
        Compute(job, job.out_dir);
        cout << "DONE:" << job.out_dir << endl;
        cout.flush();
    }
}

// ── MAIN ───────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if(argc<2) { cerr<<"Usage: chronos_dual_gpu_daemon.exe <hlsl_dir>"<<endl; return 1; }
    fs::path hlsl_dir=argv[1];

    SetEnvironmentVariableA("AmdPowerXpressRequestHighPerformance","1");
    SetEnvironmentVariableA("AMD_OPENCL_DEVICE","1");

    CreateDXGIFactory(__uuidof(IDXGIFactory),(void**)&g_factory);

    // ── Enumerate adapters: sort by VRAM ─────────────────────────────────────
    IDXGIAdapter* adp=nullptr;
    IDXGIAdapter* small_adp=nullptr;  // 780M (integrated, less VRAM)
    IDXGIAdapter* big_adp  =nullptr;  // 7700S (discrete, more VRAM)
    SIZE_T big_vram=0;
    for(UINT i=0; g_factory->EnumAdapters(i,&adp)!=DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC d; adp->GetDesc(&d);
        if(d.VendorId!=AMD_VENDOR_ID) { adp->Release(); continue; }
        if(d.DedicatedVideoMemory>big_vram) {
            if(big_adp) { small_adp=big_adp; } // demote previous big to small
            big_adp=adp; big_vram=d.DedicatedVideoMemory;
            wcerr<<L"[7700S candidate] "<<d.Description<<L" VRAM:"<<d.DedicatedVideoMemory/1024/1024<<L"MB"<<endl;
        } else {
            small_adp=adp;
            wcerr<<L"[780M  candidate] "<<d.Description<<L" VRAM:"<<d.DedicatedVideoMemory/1024/1024<<L"MB"<<endl;
        }
    }
    if(!big_adp) { cout<<"ERROR:No discrete AMD GPU found"<<endl; return 1; }

    // ── 7700S: create with SwapChain (3D program OS wake-lock) ───────────────
    g_hInst=GetModuleHandle(nullptr);
    WNDCLASSEXA wc{sizeof(WNDCLASSEXA),CS_HREDRAW|CS_VREDRAW,DummyWndProc,0,0,g_hInst,
                   nullptr,nullptr,nullptr,nullptr,"ChronosDual3D",nullptr};
    RegisterClassExA(&wc);
    g_hwnd=CreateWindowExA(0,"ChronosDual3D","Chronos Dual GPU",WS_OVERLAPPEDWINDOW,0,0,64,64,nullptr,nullptr,g_hInst,nullptr);
    DXGI_SWAP_CHAIN_DESC scd{};
    scd.BufferCount=1; scd.BufferDesc.Width=64; scd.BufferDesc.Height=64;
    scd.BufferDesc.Format=DXGI_FORMAT_R8G8B8A8_UNORM; scd.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow=g_hwnd; scd.SampleDesc.Count=1; scd.Windowed=TRUE;
    D3D_FEATURE_LEVEL fl;
    HRESULT hr=D3D11CreateDeviceAndSwapChain(big_adp,D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,nullptr,0,
                                              D3D11_SDK_VERSION,&scd,&g_7700s.swap,&g_7700s.dev,&fl,&g_7700s.ctx);
    if(FAILED(hr)) { cout<<"ERROR:7700S D3D11 init failed "<<hex<<hr<<endl; return 1; }
    g_7700s.adp=big_adp; g_7700s.name="RX 7700S";
    cerr<<"[INIT] 7700S WAKE-LOCK ACTIVE (SwapChain registered as 3D program)"<<endl;

    // ── 780M: create compute-only device (no SwapChain needed, always awake) ─
    if(small_adp) {
        D3D_FEATURE_LEVEL fl2;
        hr=D3D11CreateDevice(small_adp,D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,nullptr,0,D3D11_SDK_VERSION,
                              &g_780m.dev,&fl2,&g_780m.ctx);
        if(SUCCEEDED(hr)) { g_780m.adp=small_adp; g_780m.name="780M";
            cerr<<"[INIT] 780M compute device ready (staging GPU)"<<endl; }
    }

    // ── Compile shader once on 7700S ──────────────────────────────────────────
    fs::path hlsl=hlsl_dir/"observer_kernel.hlsl";
    ID3DBlob *code=nullptr,*err=nullptr;
    hr=D3DCompileFromFile(hlsl.wstring().c_str(),nullptr,nullptr,"ComputeConsensus","cs_5_0",0,0,&code,&err);
    if(FAILED(hr)) {
        if(err) cout<<"ERROR:"<<(char*)err->GetBufferPointer()<<endl;
        else    cout<<"ERROR:shader compile failed"<<endl; return 1;
    }
    g_7700s.dev->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&g_cs);
    code->Release();

    cout<<"READY"<<endl; cout.flush();

    // ── Launch dual threads ───────────────────────────────────────────────────
    JobQueue queue;
    thread producer(ProducerThread, ref(queue));
    thread consumer(ConsumerThread, ref(queue));
    producer.join();
    consumer.join();

    // ── Cleanup ───────────────────────────────────────────────────────────────
    g_cs->Release();
    g_7700s.ctx->Release(); g_7700s.dev->Release(); g_7700s.swap->Release(); g_7700s.adp->Release();
    if(g_780m.dev) { g_780m.ctx->Release(); g_780m.dev->Release(); g_780m.adp->Release(); }
    g_factory->Release();
    DestroyWindow(g_hwnd); UnregisterClassA("ChronosDual3D",g_hInst);
    return 0;
}
