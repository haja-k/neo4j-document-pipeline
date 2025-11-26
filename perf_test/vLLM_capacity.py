"""Test vLLM embedding endpoint capacity directly."""
import asyncio
import httpx
import time
import os
from dotenv import load_dotenv

load_dotenv()

VLLM_URL = os.getenv("SAINS_VLLM_BASE_URL", "http://localhost:8000")
VLLM_API_KEY = os.getenv("SAINS_VLLM_API_KEY", "")
MODEL = "Qwen/Qwen3-Embedding-8B"

async def test_single_embed(session: httpx.AsyncClient, text: str, req_id: int):
    """Test single embedding request."""
    start = time.perf_counter()
    try:
        resp = await session.post(
            f"{VLLM_URL}/embeddings",
            json={"model": MODEL, "input": [text]},
            timeout=120.0
        )
        duration = time.perf_counter() - start
        resp.raise_for_status()
        return {"id": req_id, "success": True, "duration": duration, "status": resp.status_code}
    except Exception as e:
        duration = time.perf_counter() - start
        return {"id": req_id, "success": False, "duration": duration, "error": str(e)}

async def test_concurrent_capacity(num_requests: int, text: str = "What is the national strategy?"):
    """Test vLLM with N concurrent embedding requests."""
    print(f"\n{'='*60}")
    print(f"Testing {num_requests} concurrent embedding requests")
    print(f"{'='*60}")
    
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
        timeout=120.0
    ) as session:
        # Fire all requests simultaneously
        start_time = time.perf_counter()
        tasks = [test_single_embed(session, text, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.perf_counter() - start_time
        
        # Analyze results
        successes = [r for r in results if isinstance(r, dict) and r.get("success")]
        failures = [r for r in results if isinstance(r, dict) and not r.get("success")]
        
        if successes:
            avg_latency = sum(r["duration"] for r in successes) / len(successes)
            p95_latency = sorted([r["duration"] for r in successes])[int(len(successes) * 0.95)]
            throughput = len(successes) / total_duration
        else:
            avg_latency = p95_latency = throughput = 0
        
        # Display results
        print(f"Total wall-clock time: {total_duration:.2f}s")
        print(f"Success rate: {len(successes)}/{num_requests} ({len(successes)/num_requests*100:.1f}%)")
        print(f"Throughput: {throughput:.2f} embeds/sec")
        print(f"Avg latency: {avg_latency*1000:.0f}ms")
        print(f"P95 latency: {p95_latency*1000:.0f}ms")
        
        if failures:
            print(f"\n⚠️  {len(failures)} failures:")
            for f in failures[:3]:  # Show first 3
                print(f"  - {f.get('error', 'Unknown error')}")
        
        return {
            "concurrent": num_requests,
            "success_rate": len(successes)/num_requests,
            "throughput": throughput,
            "avg_latency_ms": avg_latency*1000,
            "p95_latency_ms": p95_latency*1000,
            "total_time": total_duration
        }

async def run_capacity_test():
    """Run progressive load test on vLLM."""
    print(f"🔬 vLLM Embedding Capacity Test")
    print(f"Endpoint: {VLLM_URL}/embeddings")
    print(f"Model: {MODEL}")
    
    # Test with increasing load
    test_levels = [1, 2, 5, 10, 15, 20, 30, 50]
    results = []
    
    for n in test_levels:
        result = await test_concurrent_capacity(n)
        results.append(result)
        
        # Stop if success rate drops below 80%
        if result["success_rate"] < 0.8:
            print(f"\n❌ Stopping: Success rate dropped below 80%")
            break
        
        # Stop if throughput decreases (saturation point)
        if len(results) >= 2 and result["throughput"] < results[-2]["throughput"] * 0.8:
            print(f"\n⚠️  Throughput degraded significantly. Likely at capacity.")
            break
        
        await asyncio.sleep(2)  # Cooldown between tests
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 CAPACITY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"{'Concurrent':<12} {'Success%':<10} {'Throughput':<15} {'Avg Latency':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['concurrent']:<12} {r['success_rate']*100:>7.1f}%   {r['throughput']:>7.2f} req/s   {r['avg_latency_ms']:>10.0f}ms")
    
    # Find optimal capacity
    best = max(results, key=lambda x: x["throughput"])
    print(f"\n🎯 Optimal capacity: ~{best['concurrent']} concurrent requests")
    print(f"   Max throughput: {best['throughput']:.2f} embeds/sec")

if __name__ == "__main__":
    asyncio.run(run_capacity_test())