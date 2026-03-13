import json
import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from time_dilation_dlinoss_learner import (
    analyze_entangled_shared_now,
    analyze_joint_system,
    analyze_now_stream_mesh,
    build_now_stream_agents,
    build_feature_sequences,
    cluster_phase_shift_events,
    find_peak_indices,
    load_cached_packet_records,
    load_packet_records,
    make_stream_key,
    make_windows,
    normalize_sequences,
    prepare_time_stream_cache,
    summarize_canonical_oscillator_fit,
    summarize_clock_array,
    TimeMomentStreamingService,
    train_now_stream_agent_models,
)
from time_dilation_stream_dataset import load_time_stream_dataset_records


def make_packet(
    gpu_id: int,
    gpu_name: str,
    packet_index: int,
    host_time_mid: float,
    clock_values: list[int],
    pid: int | None = None,
    stream_id: str | None = None,
    device_worker_index: int | None = None,
    stream_global_index: int | None = None,
    workers_per_gpu: int | None = None,
) -> dict:
    packet = {
        "record_type": "gpu_clock_packet",
        "gpu_id": gpu_id,
        "gpu_name": gpu_name,
        "packet_index": packet_index,
        "host_time_mid": host_time_mid,
        "elapsed_host_seconds": 0.001,
        "sample_count": len(clock_values),
        "total_threads": 64,
        "buffer_size": 128,
        "clock_data": np.array(clock_values, dtype=np.uint64),
    }
    if pid is not None:
        packet["pid"] = pid
    if stream_id is not None:
        packet["stream_id"] = stream_id
    if device_worker_index is not None:
        packet["device_worker_index"] = device_worker_index
    if stream_global_index is not None:
        packet["stream_global_index"] = stream_global_index
    if workers_per_gpu is not None:
        packet["workers_per_gpu"] = workers_per_gpu
    return packet


class TimeDilationDLinOSSTests(unittest.TestCase):
    def test_summarize_clock_array(self) -> None:
        summary = summarize_clock_array([10, 12, 15, 21])
        self.assertAlmostEqual(summary["mean"], 14.5)
        self.assertEqual(summary["range"], 11.0)
        self.assertGreater(summary["diff_mean_abs"], 0.0)

    def test_build_feature_sequences_sets_peer_features(self) -> None:
        records = [
            make_packet(0, "A", 0, 1.000, [10, 11, 12]),
            make_packet(1, "B", 0, 1.004, [20, 21, 22]),
            make_packet(0, "A", 1, 1.020, [13, 14, 15]),
            make_packet(1, "B", 1, 1.024, [23, 24, 25]),
        ]
        payload = build_feature_sequences(records, bin_width=0.01, stream_granularity="gpu")
        self.assertTrue(payload["multi_processor_observed"])
        self.assertTrue(payload["multi_gpu_observed"])
        feature_names = payload["feature_names"]
        peer_idx = feature_names.index("peer_count")
        delta_idx = feature_names.index("clock_mean_vs_peers")
        first_gpu = payload["sequences"][make_stream_key(records[0], "gpu")]
        self.assertGreater(first_gpu[0, peer_idx], 0.0)
        self.assertLess(first_gpu[0, delta_idx], 0.0)

    def test_build_feature_sequences_gpu_pid_splits_same_gpu_different_pid(self) -> None:
        records = [
            make_packet(0, "A", 0, 1.000, [10, 11, 12], pid=101),
            make_packet(0, "A", 0, 1.004, [20, 21, 22], pid=202),
        ]
        payload = build_feature_sequences(records, bin_width=0.01, stream_granularity="gpu_pid")
        self.assertTrue(payload["multi_processor_observed"])
        self.assertFalse(payload["multi_gpu_observed"])
        self.assertEqual(len(payload["sequences"]), 2)
        self.assertIn("gpu_id_0_gpu_name_A_pid_101", payload["stream_labels"])

    def test_build_feature_sequences_stream_id_splits_same_gpu_same_pid(self) -> None:
        records = [
            make_packet(0, "A", 0, 1.000, [10, 11, 12], pid=101, stream_id="gpu0_worker0", device_worker_index=0, stream_global_index=0, workers_per_gpu=2),
            make_packet(0, "A", 0, 1.004, [20, 21, 22], pid=101, stream_id="gpu0_worker1", device_worker_index=1, stream_global_index=1, workers_per_gpu=2),
        ]
        payload = build_feature_sequences(records, bin_width=0.01, stream_granularity="stream_id")
        self.assertTrue(payload["multi_processor_observed"])
        self.assertFalse(payload["multi_gpu_observed"])
        self.assertEqual(len(payload["sequences"]), 2)
        self.assertIn("stream_id_gpu0_worker0", payload["stream_labels"])
        first_key = make_stream_key(records[0], "stream_id")
        self.assertEqual(payload["stream_info"][first_key]["device_worker_index"], 0)
        self.assertEqual(payload["stream_info"][first_key]["workers_per_gpu"], 2)

    def test_make_stream_key_stream_id_falls_back_to_worker_identity(self) -> None:
        record = make_packet(2, "C", 0, 1.000, [10, 11], pid=88, device_worker_index=3)
        key = make_stream_key(record, "stream_id")
        self.assertEqual(dict(key)["stream_id"], "gpu2_worker3")

    def test_load_time_stream_dataset_records_preserves_explicit_stream_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "stream_dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema_version": 1,
                "dataset_type": "gpu_clock_time_stream_dataset",
                "packet_count": 2,
                "chunk_files": ["chunk_00000.npz"],
                "workers_per_gpu": 2,
                "stream_count": 2,
            }
            (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.savez(
                dataset_dir / "chunk_00000.npz",
                gpu_id=np.asarray([0, 0], dtype=np.int32),
                gpu_name=np.asarray(["A", "A"]),
                pid=np.asarray([101, 202], dtype=np.int64),
                stream_id=np.asarray(["gpu0_worker0", "gpu0_worker1"]),
                device_worker_index=np.asarray([0, 1], dtype=np.int32),
                stream_global_index=np.asarray([0, 1], dtype=np.int32),
                workers_per_gpu=np.asarray([2, 2], dtype=np.int32),
                packet_index=np.asarray([0, 0], dtype=np.int64),
                host_time_start=np.asarray([1.0, 1.1], dtype=np.float64),
                host_time_end=np.asarray([1.01, 1.11], dtype=np.float64),
                host_time_mid=np.asarray([1.005, 1.105], dtype=np.float64),
                elapsed_host_seconds=np.asarray([0.01, 0.01], dtype=np.float64),
                threads_per_block=np.asarray([64, 64], dtype=np.int32),
                blocks=np.asarray([2, 2], dtype=np.int32),
                samples_per_thread=np.asarray([4, 4], dtype=np.int32),
                total_threads=np.asarray([128, 128], dtype=np.int32),
                buffer_size=np.asarray([512, 512], dtype=np.int32),
                sample_count=np.asarray([3, 3], dtype=np.int32),
                clock_data_offset=np.asarray([0, 3, 6], dtype=np.int64),
                clock_data_flat=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.uint64),
            )
            records, loaded_manifest = load_time_stream_dataset_records(dataset_dir)
        self.assertEqual(loaded_manifest["stream_count"], 2)
        self.assertEqual(records[0]["stream_id"], "gpu0_worker0")
        self.assertEqual(records[1]["device_worker_index"], 1)
        self.assertEqual(records[1]["stream_global_index"], 1)
        self.assertEqual(records[0]["workers_per_gpu"], 2)

    def test_load_time_stream_dataset_records_backfills_missing_stream_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "stream_dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema_version": 1,
                "dataset_type": "gpu_clock_time_stream_dataset",
                "packet_count": 1,
                "chunk_files": ["chunk_00000.npz"],
            }
            (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            np.savez(
                dataset_dir / "chunk_00000.npz",
                gpu_id=np.asarray([0], dtype=np.int32),
                gpu_name=np.asarray(["A"]),
                pid=np.asarray([101], dtype=np.int64),
                packet_index=np.asarray([0], dtype=np.int64),
                host_time_start=np.asarray([1.0], dtype=np.float64),
                host_time_end=np.asarray([1.01], dtype=np.float64),
                host_time_mid=np.asarray([1.005], dtype=np.float64),
                elapsed_host_seconds=np.asarray([0.01], dtype=np.float64),
                threads_per_block=np.asarray([64], dtype=np.int32),
                blocks=np.asarray([2], dtype=np.int32),
                samples_per_thread=np.asarray([4], dtype=np.int32),
                total_threads=np.asarray([128], dtype=np.int32),
                buffer_size=np.asarray([512], dtype=np.int32),
                sample_count=np.asarray([3], dtype=np.int32),
                clock_data_offset=np.asarray([0, 3], dtype=np.int64),
                clock_data_flat=np.asarray([1, 2, 3], dtype=np.uint64),
            )
            records, _ = load_time_stream_dataset_records(dataset_dir)
        self.assertEqual(records[0]["stream_id"], "gpu0_pid101")
        self.assertEqual(records[0]["device_worker_index"], -1)
        self.assertEqual(records[0]["workers_per_gpu"], 1)

    def test_prepare_time_stream_cache_round_trips_filtered_records(self) -> None:
        source_records = [
            make_packet(0, "A", 1, 1.020, [13, 14, 15]),
            {"record_type": "other", "ignored": True},
            make_packet(1, "B", 0, 1.004, [20, 21, 22], pid=202),
            make_packet(0, "A", 0, 1.000, [10, 11, 12], pid=101),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "packets.npy"
            cache_dir = Path(tmpdir) / "time_stream_cache"
            np.save(input_path, np.array(source_records, dtype=object), allow_pickle=True)
            raw_records = load_packet_records(input_path)
            manifest = prepare_time_stream_cache(input_path, cache_dir, chunk_size=2)
            cached_records, cached_manifest = load_cached_packet_records(input_path, cache_dir)
        self.assertIn(manifest["cache_status"], {"created", "rebuilt", "reused"})
        self.assertEqual(manifest["record_count"], len(raw_records))
        self.assertEqual(cached_manifest["record_count"], len(raw_records))
        self.assertEqual(len(cached_records), len(raw_records))
        for raw_record, cached_record in zip(raw_records, cached_records):
            self.assertEqual(
                {k: v for k, v in raw_record.items() if k != "clock_data"},
                {k: v for k, v in cached_record.items() if k != "clock_data"},
            )
            np.testing.assert_array_equal(raw_record["clock_data"], cached_record["clock_data"])

    def test_time_stream_service_matches_batch_feature_sequences(self) -> None:
        records = [
            make_packet(1, "B", 0, 1.004, [20, 21, 22], pid=202),
            make_packet(0, "A", 0, 1.000, [10, 11, 12], pid=101),
            make_packet(0, "A", 1, 1.020, [13, 14, 15], pid=101),
            make_packet(1, "B", 1, 1.024, [23, 24, 25], pid=202),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "packets.npy"
            cache_dir = Path(tmpdir) / "time_stream_cache"
            np.save(input_path, np.array(records, dtype=object), allow_pickle=True)
            batch_records = load_packet_records(input_path)
            prepare_time_stream_cache(input_path, cache_dir, chunk_size=2)
            cached_records, _ = load_cached_packet_records(input_path, cache_dir)
        batch_payload = build_feature_sequences(batch_records, bin_width=0.01, stream_granularity="gpu_pid")
        cached_payload = build_feature_sequences(cached_records, bin_width=0.01, stream_granularity="gpu_pid")
        service = TimeMomentStreamingService(cached_records, bin_width=0.01)
        moments = list(service.iter_moments())
        self.assertEqual([moment["time_bin"] for moment in moments], [0, 2])
        self.assertEqual(batch_payload["feature_names"], cached_payload["feature_names"])
        self.assertEqual(batch_payload["stream_labels"], cached_payload["stream_labels"])
        self.assertEqual(batch_payload["global_start"], cached_payload["global_start"])
        for key in batch_payload["sequences"]:
            np.testing.assert_allclose(batch_payload["sequences"][key], cached_payload["sequences"][key])
            self.assertEqual(batch_payload["metadata"][key], cached_payload["metadata"][key])

    def test_make_windows(self) -> None:
        sequence = np.arange(20, dtype=np.float32).reshape(10, 2)
        inputs, targets = make_windows({(("gpu_id", 0), ("gpu_name", "A")): sequence}, window_size=5, stride=2)
        self.assertEqual(inputs.shape, (3, 4, 2))
        self.assertEqual(targets.shape, (3, 4, 2))

    def test_find_peak_indices_respects_spacing(self) -> None:
        scores = np.array([0.0, 1.0, 5.0, 2.0, 0.0, 4.5, 1.0, 0.0], dtype=np.float32)
        peaks, z_scores = find_peak_indices(scores, z_threshold=0.5, min_separation=3, max_events=4, fallback_top_k=4)
        self.assertEqual(peaks, [2, 5])
        self.assertGreater(z_scores[2], 0.5)

    def test_cluster_phase_shift_events_finds_repeat(self) -> None:
        events = [
            {
                "processor_label": "gpu_0_A",
                "transition_index": 3,
                "packet_index": 3,
                "host_time_mid": 1.0,
                "distortion_score": 2.0,
                "distortion_score_z": 3.0,
                "signature": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            },
            {
                "processor_label": "gpu_0_A",
                "transition_index": 9,
                "packet_index": 9,
                "host_time_mid": 2.0,
                "distortion_score": 1.8,
                "distortion_score_z": 2.8,
                "signature": np.array([0.99, 0.01, 0.0], dtype=np.float32),
            },
            {
                "processor_label": "gpu_0_A",
                "transition_index": 12,
                "packet_index": 12,
                "host_time_mid": 3.0,
                "distortion_score": 1.1,
                "distortion_score_z": 1.5,
                "signature": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            },
        ]
        clustered = cluster_phase_shift_events(events, similarity_threshold=0.95)
        self.assertEqual(clustered["class_count"], 2)
        self.assertEqual(clustered["repeated_class_count"], 1)
        self.assertEqual(clustered["repeated_event_count"], 2)

    def test_analyze_joint_system_fits_operator_for_aligned_streams(self) -> None:
        records = [
            make_packet(0, "A", 0, 1.000, [10, 11, 12]),
            make_packet(1, "B", 0, 1.004, [20, 21, 22]),
            make_packet(0, "A", 1, 1.020, [13, 14, 15]),
            make_packet(1, "B", 1, 1.024, [23, 24, 25]),
            make_packet(0, "A", 2, 1.040, [16, 17, 18]),
            make_packet(1, "B", 2, 1.044, [26, 27, 28]),
        ]
        payload = build_feature_sequences(records, bin_width=0.01, stream_granularity="gpu")
        normalized, _, _ = normalize_sequences(payload["sequences"])
        summary, arrays = analyze_joint_system(
            payload["sequences"],
            normalized,
            payload["metadata"],
            payload["feature_names"],
            ridge=1e-3,
        )
        self.assertEqual(summary["stream_count"], 2)
        self.assertEqual(summary["aligned_bin_count"], 3)
        self.assertEqual(summary["complete_bin_count"], 3)
        self.assertEqual(summary["operator_fit"]["status"], "ok")
        self.assertIn("empirical_operator", arrays)
        self.assertEqual(arrays["empirical_operator"].shape[0], 2 * len(payload["feature_names"]))
        self.assertEqual(len(arrays["decoherence_score"]), 3)

    def test_analyze_entangled_shared_now_tracks_memory_and_operator(self) -> None:
        records = [
            make_packet(0, "A", 0, 1.000, [10, 11, 12]),
            make_packet(1, "B", 0, 1.004, [20, 21, 22]),
            make_packet(0, "A", 1, 1.020, [16, 17, 18]),
            make_packet(1, "B", 1, 1.024, [24, 25, 26]),
            make_packet(0, "A", 2, 1.040, [13, 14, 15]),
            make_packet(1, "B", 2, 1.044, [30, 31, 32]),
        ]
        payload = build_feature_sequences(records, bin_width=0.01, stream_granularity="gpu")
        normalized, _, _ = normalize_sequences(payload["sequences"])
        summary, arrays, event_payload = analyze_entangled_shared_now(
            payload["sequences"],
            normalized,
            payload["metadata"],
            ridge=1e-3,
            lattice_size=5,
            memory_mix=0.25,
            event_z_threshold=10.0,
            event_min_separation=1,
            event_top_k=2,
            canonical_pairing="half_split",
        )
        self.assertEqual(summary["stream_count"], 2)
        self.assertEqual(summary["aligned_bin_count"], 3)
        self.assertEqual(summary["memory_lattice_size"], 5)
        self.assertEqual(summary["operator_fit"]["status"], "ok")
        self.assertEqual(summary["hamiltonian_qp_analysis"]["status"], "ok")
        self.assertIn("orthogonal", summary["methodology_comparison"])
        self.assertIn("symplectic", summary["methodology_comparison"])
        self.assertIn("Markovian latent", summary["projection"])
        self.assertEqual(arrays["shared_now_state_matrix"].shape, (3, len(payload["feature_names"])))
        self.assertEqual(arrays["shared_now_empirical_operator"].shape, (len(payload["feature_names"]), len(payload["feature_names"])))
        self.assertEqual(arrays["shared_now_canonical_state_matrix"].shape[0], 3)
        self.assertIn("shared_now_canonical_symplectic_operator", arrays)
        self.assertAlmostEqual(float(arrays["memory_context_norm"][0]), 0.0, places=6)
        self.assertGreater(float(arrays["memory_delta"][0]), 0.0)
        self.assertGreaterEqual(len(event_payload["events"]), 1)
        self.assertLessEqual(len(event_payload["events"]), 2)
        self.assertEqual(event_payload["parameters"]["top_k"], 2)
        for label in payload["stream_labels"]:
            self.assertIn(label, summary["per_stream_mean_shared_now_deviation"])
            self.assertIn(f"shared_now_deviation_{label}", arrays)

    def test_build_now_stream_agents_creates_temporal_views(self) -> None:
        records = [
            make_packet(0, "A", 0, 1.000, [10, 11, 12]),
            make_packet(0, "A", 1, 1.020, [13, 14, 15]),
            make_packet(0, "A", 2, 1.040, [16, 17, 18]),
        ]
        payload = build_feature_sequences(records, bin_width=0.01, stream_granularity="gpu")
        normalized, _, _ = normalize_sequences(payload["sequences"])
        agent_payload = build_now_stream_agents(payload["sequences"], normalized, payload["metadata"], [0, 1])
        self.assertEqual(agent_payload["base_stream_count"], 1)
        self.assertEqual(agent_payload["agent_count"], 2)
        self.assertEqual(agent_payload["temporal_radii"], [0, 1])
        radius_one_key = next(key for key in agent_payload["metadata"] if dict(key).get("now_scale_radius") == 1)
        first_row = agent_payload["metadata"][radius_one_key][0]
        self.assertAlmostEqual(first_row["agent_believed_now_host_time_mid"], (1.000 + 1.020) / 2.0)
        self.assertAlmostEqual(first_row["agent_now_offset_seconds"], 0.010)
        self.assertEqual(first_row["source_stream_label"], next(iter(payload["stream_labels"])))

    def test_train_now_stream_agent_models_marks_insufficient_packets(self) -> None:
        key = (("gpu_id", 0), ("gpu_name", "A"), ("now_agent_kind", "temporal_scale"), ("now_scale_radius", 0))
        summaries, details = train_now_stream_agent_models(
            {key: np.ones((3, 2), dtype=np.float32)},
            {key: [{"packet_index": i, "time_bin": i, "host_time_mid": float(i), "elapsed_host_seconds": float(i)} for i in range(3)]},
            argparse.Namespace(
                window_size=5,
                stride=1,
                state_dim=4,
                hidden_dim=4,
                num_blocks=1,
                learning_rate=1e-3,
                epochs=1,
                batch_size=2,
                qp_regularization_weight=0.0,
                energy_drift_regularization_weight=0.0,
                spring_regularization_weight=0.0,
                canonical_pairing="half_split",
            ),
            "cpu",
        )
        self.assertEqual(details, {})
        label = next(iter(summaries))
        self.assertEqual(summaries[label]["status"], "insufficient_packets")
        self.assertEqual(summaries[label]["window_size"], 5)

    def test_analyze_now_stream_mesh_tracks_shared_qpc_and_agents(self) -> None:
        records = [
            make_packet(0, "A", 0, 1.000, [10, 11, 12]),
            make_packet(1, "B", 0, 1.004, [20, 21, 22]),
            make_packet(0, "A", 1, 1.020, [14, 15, 16]),
            make_packet(1, "B", 1, 1.024, [24, 25, 26]),
            make_packet(0, "A", 2, 1.040, [18, 19, 20]),
            make_packet(1, "B", 2, 1.044, [28, 29, 30]),
        ]
        payload = build_feature_sequences(records, bin_width=0.01, stream_granularity="gpu")
        normalized, _, _ = normalize_sequences(payload["sequences"])
        summary, arrays, event_payload = analyze_now_stream_mesh(
            payload["sequences"],
            normalized,
            payload["metadata"],
            temporal_radii=[0, 1],
            ridge=1e-3,
            lattice_size=5,
            memory_mix=0.25,
            event_z_threshold=10.0,
            event_min_separation=1,
            event_top_k=2,
            canonical_pairing="half_split",
        )
        feature_count = len(payload["feature_names"])
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["base_stream_count"], 2)
        self.assertEqual(summary["agent_count"], 4)
        self.assertEqual(summary["aligned_bin_count"], 3)
        self.assertEqual(summary["shared_qpc_operator_fit"]["status"], "ok")
        self.assertEqual(summary["mesh_operator_fit"]["status"], "ok")
        self.assertEqual(len(summary["agent_manifest"]), 4)
        self.assertEqual(len(summary["per_agent_mean_qpc_coherence"]), 4)
        self.assertIn("Gregorian host_time_mid", summary["projection"])
        self.assertEqual(arrays["shared_qpc_now_state_matrix"].shape, (3, feature_count))
        self.assertEqual(arrays["shared_qpc_memory_estimate_matrix"].shape, (3, feature_count))
        self.assertEqual(arrays["complete_mesh_state_matrix"].shape, (3, 4 * feature_count))
        self.assertEqual(event_payload["parameters"]["top_k"], 2)
        self.assertIn("events", event_payload)
        self.assertIn("classes", event_payload)

    def test_summarize_canonical_oscillator_fit_reports_symplectic_diagnostics(self) -> None:
        theta = np.linspace(0.0, np.pi / 2.0, 8, dtype=np.float32)
        state_matrix = np.stack([
            np.cos(theta),
            np.sin(theta),
        ], axis=1).astype(np.float32)
        summary, arrays = summarize_canonical_oscillator_fit(state_matrix, ridge=1e-4)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["canonical_pair_count"], 1)
        self.assertIn(summary["preferred_reference"], {"orthogonal", "symplectic"})
        self.assertLessEqual(summary["projected_symplectic_defect"], summary["empirical_operator_symplectic_defect"] + 1e-5)
        self.assertLess(summary["canonical_energy_drift_mean_abs"], 0.1)
        self.assertEqual(arrays["canonical_state_matrix"].shape, state_matrix.shape)
        self.assertEqual(arrays["canonical_q_matrix"].shape, (len(theta), 1))
        self.assertEqual(arrays["canonical_p_matrix"].shape, (len(theta), 1))
        self.assertEqual(arrays["canonical_symplectic_operator"].shape, (2, 2))


if __name__ == "__main__":
    unittest.main()