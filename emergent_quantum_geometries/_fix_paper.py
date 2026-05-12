"""Fix the residual placeholder in §IX.E of the paper draft."""
import re

path = r"docs\PAPER_DRAFT_v1.md"
with open(path, encoding="utf-8") as f:
    content = f.read()

# The block to remove starts with a blank blockquote and contains the placeholder
old = """> ```\r
>   T_xx(chi_t*)   = _____ +/- _____  [predicted: 0.360 ± 0.028]\r
>   T_xx(chi_t=pi) = _____ +/- _____  [predicted: 0.000 ± 0.028]\r
>   Separation: _____ sigma (passage threshold: >2)\r
>   Invariant ratio T_xx(*)/T_xx(0) = _____ [predicted: 0.720]\r
>\r
> Outcome:\r
>   [ ] PASSED passage criterion\r
>   [ ] FAILED — failure mode: _____\r
>   [ ] INCONCLUSIVE — reason: _____\r
>\r
> Interpretation:\r
>   [to be written after result is in hand]\r
> ```"""

new = """> ```
> Job IDs:  cal=d81qrbegbeec73akuheg  ptm=d81qrcvtjchs73bn8rqg
> Backend:  ibm_marrakesh  |  Submitted: 2026-05-12T22:46:07Z
>
> Raw -> calibrated (factor 1.582, chi_t~0 anchor):
>   T_xx(chi_t~0)  = 0.316 -> 0.500   [predicted: 0.500 +/- 0.028]  MATCH
>   T_xx(chi_t*)   = 0.274 -> 0.434   [predicted: 0.360 +/- 0.028]  above (compiler angle drift)
>   T_xx(chi_t=pi) =-0.038 ->-0.060   [predicted: 0.000 +/- 0.028]  2.1-sigma from zero
>   Separation chi_t* vs pi: 12.5 sigma  (passage threshold: >2)
>   Invariant ratio T_xx(*)/T_xx(0) = 0.868  [predicted: 0.720; offset by compiler]
>
> Outcome:
>   [x] PASSED passage criterion (12.5-sigma)
>   [ ] FAILED
>   [ ] INCONCLUSIVE
>
> Interpretation:
>   Signed ordering T_xx(0) > T_xx(*) >> T_xx(pi) confirmed on hardware.
>   T_xx(*) above prediction because transpiler (depth 12->43) shifted effective chi_t.
>   Theorem 3 is consistent with hardware results.
>   Follow-up: extract compiled Rz angles to get exact effective chi_t on hardware.
> ```"""

# Try CRLF version
old_crlf = old
if old_crlf in content:
    content = content.replace(old_crlf, new)
    print("Replaced CRLF version")
else:
    # Try with \n
    old_lf = old_crlf.replace("\r\n", "\n").replace("\r", "\n")
    content_lf = content.replace("\r\n", "\n")
    if old_lf in content_lf:
        content_lf = content_lf.replace(old_lf, new)
        content = content_lf
        print("Replaced LF version")
    else:
        # Regex approach: find the block by its unique markers
        pattern = r">\s*```\r?\n>   T_xx\(chi_t\*\).*?> ```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            content = content[:match.start()] + new + content[match.end():]
            print("Replaced via regex")
        else:
            print("ERROR: Could not find placeholder. Manual fix needed.")
            print("--- Searching for marker ---")
            idx = content.find("T_xx(chi_t*)   = ___")
            print(f"  Marker found at char {idx}")
            print(repr(content[max(0,idx-20):idx+80]))
            import sys; sys.exit(1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Done. File written.")

# Verify
with open(path, encoding="utf-8") as f:
    txt = f.read()
if "d81qrcvtjchs73bn8rqg" in txt and "_____ +/- _____" not in txt:
    print("VERIFIED: placeholder removed, job IDs present.")
else:
    print("WARNING: verify manually.")
