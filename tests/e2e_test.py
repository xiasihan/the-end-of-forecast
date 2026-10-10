"""E2E test for the full platform — create project, run pipeline, verify results."""
import requests
import json
import time
import sys
import os

BASE = "http://127.0.0.1:8000/api"

def test_health():
    r = requests.get(f"{BASE}/health")
    assert r.json()["status"] == "ok", f"Health failed: {r.text}"
    print("✅ Health OK")

def test_create_project():
    with open("tests/fixtures/anhui_spread.csv", "rb") as f:
        r = requests.post(
            f"{BASE}/projects",
            data={
                "name": "E2E Final Test",
                "description": "Predict Anhui spread direction: up or down",
                "target_col": "y",
                "time_col": "ds",
                "budget": "fast",
            },
            files={"file": ("anhui_spread.csv", f, "text/csv")},
        )
    assert r.status_code == 200, f"Create project failed: {r.status_code} {r.text}"
    data = r.json()
    assert data["id"].startswith("proj_"), f"Bad project ID: {data['id']}"
    print(f"✅ Created project: {data['id']}")
    return data["id"]

def test_list_projects(proj_id):
    r = requests.get(f"{BASE}/projects")
    assert r.status_code == 200
    projects = r.json()["projects"]
    ids = [p["id"] for p in projects]
    assert proj_id in ids, f"Project {proj_id} not in list: {ids}"
    print(f"✅ Listed {len(projects)} projects, found target")

def test_start_run(proj_id):
    r = requests.post(f"{BASE}/projects/{proj_id}/runs?budget=fast")
    assert r.status_code == 200, f"Start run failed: {r.status_code} {r.text}"
    data = r.json()
    assert data["status"] == "running"
    print(f"✅ Started run: {data['run_id']}")
    return data["run_id"]

def wait_for_completion(proj_id, run_id, max_wait=180):
    start = time.time()
    while time.time() - start < max_wait:
        r = requests.get(f"{BASE}/projects/{proj_id}/runs")
        assert r.status_code == 200, f"List runs failed: {r.status_code}"
        runs = r.json().get("runs", [])
        if not runs:
            time.sleep(3)
            continue
        status = runs[0].get("status", "unknown")
        elapsed = runs[0].get("elapsed_seconds", "?")
        elapsed_str = f"{elapsed:.0f}s" if isinstance(elapsed, (int, float)) else str(elapsed)
        print(f"  Status: {status} ({elapsed_str})")
        if status in ("completed", "failed"):
            return status
        time.sleep(5)
    return "timeout"

def test_get_result(proj_id, run_id):
    r = requests.get(f"{BASE}/projects/{proj_id}/runs/{run_id}")
    assert r.status_code == 200, f"Get run failed: {r.status_code} {r.text}"
    data = r.json()
    print(f"\n=== Pipeline Result ===")
    print(data.get("summary", "")[:500])
    result = data.get("result", {})
    if result:
        mt = result.get("metric_tiers", {})
        print(f"Best model: {result.get('best_model')} ({result.get('best_metric_name')}={result.get('best_metric_value')})")
        print(f"Holdout: {mt.get('holdout_score')}")
        print(f"Validation: {mt.get('validation_score')}")
        print(f"Refinement: {mt.get('refinement_score')}")
        print(f"Ensemble: {mt.get('ensemble_score')}")
    return data

def test_sse_stream(proj_id, run_id):
    """Test SSE endpoint delivers events."""
    r = requests.get(
        f"{BASE}/projects/{proj_id}/runs/{run_id}/stream",
        stream=True,
        headers={"Accept": "text/event-stream"},
        timeout=10,
    )
    lines = []
    for line in r.iter_lines(decode_unicode=True):
        if line:
            lines.append(line)
        if len(lines) >= 5:
            break
    print(f"✅ SSE stream: got {len(lines)} lines")
    for line in lines[:5]:
        print(f"  {line[:100]}")

def test_api_keys():
    r = requests.post(f"{BASE}/keys", json={"name": "e2e-test-key"})
    assert r.status_code == 200, f"Create key failed: {r.status_code} {r.text}"
    data = r.json()
    assert "raw_key" in data
    print(f"✅ Created API key: {data['prefix']}")

    r = requests.get(f"{BASE}/keys")
    assert r.status_code == 200
    keys = r.json()["keys"]
    print(f"✅ Listed {len(keys)} API keys")

def main():
    test_health()
    proj_id = test_create_project()
    test_list_projects(proj_id)
    run_id = test_start_run(proj_id)
    
    status = wait_for_completion(proj_id, run_id)
    assert status == "completed", f"Pipeline did not complete: {status}"
    
    test_get_result(proj_id, run_id)
    test_sse_stream(proj_id, run_id)
    test_api_keys()
    
    # Cleanup
    requests.delete(f"{BASE}/projects/{proj_id}")
    print(f"\n🎉 ALL E2E TESTS PASSED")

if __name__ == "__main__":
    main()