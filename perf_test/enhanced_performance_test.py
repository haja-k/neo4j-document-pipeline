"""
Enhanced Performance Test for GraphRAG API
===========================================
Features:
- Smooth incremental load testing with configurable ramp-up
- Real-time queue monitoring
- Detailed metrics collection (latency, throughput, success rate)
- Rich Excel reports with charts and statistics
- Better error handling and retry logic
- Progress tracking and live statistics
"""

import requests
import time
import json
import threading
import concurrent.futures
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass, asdict
import statistics
import sys
import random
from pathlib import Path

# Rich console output (optional - falls back to print if not installed)
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.table import Table
    from rich.live import Live
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None
    print("Tip: Install 'rich' for better console output: pip install rich")

# ========================================
# CONFIGURATION
# ========================================

class TestConfig:
    """Centralized test configuration"""
    API_URL = "https://knowledge-graph.sains.com.my/graphrag"
    QUEUE_STATUS_URL = "https://knowledge-graph.sains.com.my/queue_status"
    
    # Output configuration
    OUTPUT_DIR = Path("test_results")  # Directory to save test results
    
    # Load test parameters - Find breaking point mode
    MIN_USERS = 10           # Start at 10 users
    MAX_USERS = 200          # Test up to 200 users (will stop early if API dies)
    USERS_INCREMENT = 5      # Jump by 5 users
    WAIT_BETWEEN_STEPS = 3   # Seconds to wait between load steps
    
    # Breaking point detection
    FAILURE_THRESHOLD = 50   # Stop if failure rate exceeds 50%
    CONSECUTIVE_FAILURES = 2 # Stop after 2 consecutive high-failure steps
    MIN_SUCCESS_RATE = 50    # Minimum acceptable success rate (%)
    
    # Request parameters
    REQUEST_TIMEOUT = 400  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    # Success criteria (deprecated - using FAILURE_THRESHOLD now)
    MAX_ACCEPTABLE_LATENCY_MS = 30000  # 30 seconds
    MAX_ACCEPTABLE_FAIL_RATE = 50  # Use FAILURE_THRESHOLD instead
    
    # DEBUGGING OPTIONS
    DEBUG_MODE = True       # Enable detailed logging
    LOG_REQUEST_TIMING = True  # Log when each request starts/ends
    MONITOR_QUEUE_CONTINUOUSLY = True  # Check queue during test


# Test questions
QUESTIONS = [
    "Could you please explain how the mining sector is expected to contribute to inclusivity, employment opportunities, and the development of rural communities in Sarawak?",
    "Can you explain how the handling daily operation of the facility is planned for tourism facilities development?",
    "Can u tell me how state-owned universties help with UP-DLP Sarawak?",
    "Apakah tindakan Jabatan Keselamatan dan Kesihatan Pekerjaan terhadap syarikat yang terlibat dalam bahaya utama di SIP?",
    "Boleh jelaskan berapa projek yang telah dilaksanakan oleh Sr Aman Development Agency (SADA) dan jumlah peruntukan yang diterima untuk tahun 2025?",
    "Can you tell me what are the plans for Totally Protected Areas facilities upgrade?",
    "Could you please explain the main strategies and initiatives outlined for the tourism sector in Sarawak?",
    "Can you explain how the Rajah Brooke dynasty influenced the cultural and historical development of the Sarawak Delta Geopark?",
    "Can you explain the repayment terms for an advance to purchase a new vehicle based on the Sarawak General Order 1996?",
    "Can you provide detailed information on how the projects under the tourism sector ensure timely completion with minimal delays?",
    "Datuk Seri Alexander Nanta Linggi tu dia kerja apa dalam Kabinet Malaysia sekarang?",
    "Can you explain in detail the strategies and expected outcomes of the Sarawak Heritage Ordinance administration?",
    "What are the key details and benefits of the Sarawak tourism promotion incentives?",
    "Can you provide details on the initiative for Securing Business Events for Miri?",
    "How is Sarawak planning to enhance food production for export?",
    "What are the key targets and expected economic impacts of the Business Events 2021 to 2025 initiatives in Sarawak?",
    "Could you please explain how the manufacturing sector's initiatives will benefit rural communities in Sarawak?",
    "Can you tell me what facilities will be developed at Limbang Mangrove National Park?",
    "How can I obtain permission for use of content from the PCDS 2030 Highlights 2023 document?",
    "Who is the State Secretary of Sarawak based on Cabinet Members of Malaysia and Sarawak Government?"
]


# ========================================
# DATA MODELS
# ========================================

@dataclass
class RequestResult:
    """Individual request result"""
    timestamp: str
    step: int
    user_index: int
    question: str
    status: str  # success, failed, timeout, error
    latency_ms: Optional[float]
    response_time_ms: Optional[float]
    ttft_ms: Optional[float]  # Time to first token (if applicable)
    status_code: Optional[int]
    success: bool
    error_message: str
    retry_count: int
    queue_position: Optional[int]
    request_start_time: Optional[float] = None  # NEW: Track when request actually started
    request_end_time: Optional[float] = None    # NEW: Track when request ended
    barrier_wait_time_ms: Optional[float] = None  # NEW: How long thread waited at barrier
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepMetrics:
    """Aggregated metrics for a load step"""
    step: int
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    fail_rate: float
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_rps: float
    duration_seconds: float
    errors: Dict[str, int]
    # NEW: Concurrency verification
    barrier_wait_time_ms: Optional[float] = None
    actual_concurrent_starts: int = 0  # How many requests started within 100ms of each other
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert errors dict to string for Excel
        result['errors'] = json.dumps(self.errors)
        return result


# ========================================
# API CLIENT
# ========================================

class GraphRAGClient:
    """Client for interacting with GraphRAG API"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.session = requests.Session()
        
    def get_queue_status(self) -> Optional[Dict[str, Any]]:
        """Get current queue status from API"""
        try:
            response = self.session.get(
                self.config.QUEUE_STATUS_URL,
                timeout=5
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[yellow]Warning: Could not fetch queue status: {e}[/yellow]")
            else:
                print(f"Warning: Could not fetch queue status: {e}")
        return None
    
    def send_request(self, question: str, step: int, user_index: int, start_barrier: threading.Barrier = None) -> RequestResult:
        """Send a single GraphRAG request with retry logic"""
        
        # Wait for all threads to be ready before starting (for true concurrency)
        barrier_wait_start = time.time()
        if start_barrier:
            try:
                if self.config.DEBUG_MODE:
                    if RICH_AVAILABLE:
                        console.print(f"[dim blue]User {user_index}: Waiting at barrier...[/dim blue]")
                    else:
                        print(f"User {user_index}: Waiting at barrier...")
                start_barrier.wait(timeout=30)
            except threading.BrokenBarrierError:
                pass
        barrier_wait_time = (time.time() - barrier_wait_start) * 1000
        
        # RECORD THE EXACT START TIME
        request_start_time = time.time()
        
        if self.config.DEBUG_MODE and self.config.LOG_REQUEST_TIMING:
            timestamp_str = datetime.fromtimestamp(request_start_time).strftime("%H:%M:%S.%f")[:-3]
            if RICH_AVAILABLE:
                console.print(f"[bright_blue]🚀 User {user_index}: Request STARTED at {timestamp_str}[/bright_blue]")
            else:
                print(f"🚀 User {user_index}: Request STARTED at {timestamp_str}")
        
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")
        
        result = RequestResult(
            timestamp=timestamp,
            step=step,
            user_index=user_index,
            question=question[:100],  # Truncate for readability
            status="pending",
            latency_ms=None,
            response_time_ms=None,
            ttft_ms=None,
            status_code=None,
            success=False,
            error_message="",
            retry_count=0,
            queue_position=None,
            request_start_time=request_start_time,
            request_end_time=None,
            barrier_wait_time_ms=barrier_wait_time
        )
        
        headers = {"Content-Type": "application/json"}
        payload = {"question": question}
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                response = self.session.post(
                    self.config.API_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.config.REQUEST_TIMEOUT
                )
                
                response_time_ms = (time.time() - request_start_time) * 1000
                result.response_time_ms = response_time_ms
                result.status_code = response.status_code
                result.retry_count = attempt
                
                # Check HTTP status code first
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # CRITICAL: Check API's success field
                        api_success = data.get("success", False)
                        result.success = api_success
                        result.latency_ms = response_time_ms
                        
                        if api_success:
                            result.status = "success"
                            # Extract timing info if available
                            timings = data.get("timings", {})
                            if timings:
                                result.ttft_ms = timings.get("embed", None)
                            result.request_end_time = time.time()
                            return result
                        else:
                            # API returned success: false
                            result.status = "api_failure"
                            result.error_message = data.get("message") or data.get("error_details") or "API returned success: false"
                            result.request_end_time = time.time()
                            # Don't retry API failures - they're likely valid "no data" responses
                            return result
                    except json.JSONDecodeError:
                        result.status = "invalid_json"
                        result.error_message = f"Invalid JSON response: {response.text[:200]}"
                        result.request_end_time = time.time()
                        return result
                else:
                    # HTTP error
                    result.status = "http_error"
                    result.error_message = f"HTTP {response.status_code}: {response.text[:100]}"
                    result.latency_ms = response_time_ms
                    result.request_end_time = time.time()
                    
                # If not successful and we have retries left, try again
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY * (attempt + 1))
                    continue
                else:
                    return result
                    
            except requests.exceptions.Timeout:
                result.status = "timeout"
                result.error_message = f"Request timeout after {self.config.REQUEST_TIMEOUT}s"
                result.retry_count = attempt
                result.request_end_time = time.time()
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY * (attempt + 1))
                    continue
                return result
                
            except requests.exceptions.ConnectionError as e:
                result.status = "connection_error"
                result.error_message = f"Connection error: {str(e)[:100]}"
                result.retry_count = attempt
                result.request_end_time = time.time()
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY * (2 ** attempt))
                    continue
                return result
                
            except Exception as e:
                result.status = "error"
                result.error_message = f"Unexpected error: {str(e)[:100]}"
                result.retry_count = attempt
                result.request_end_time = time.time()
                return result
        
        return result
    
    def close(self):
        """Close the session"""
        self.session.close()


# ========================================
# TEST EXECUTOR
# ========================================

class LoadTestExecutor:
    """Executes load tests and collects results"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.client = GraphRAGClient(config)
        self.all_results: List[RequestResult] = []
        self.step_metrics: List[StepMetrics] = []
        self.consecutive_high_failures = 0  # Track consecutive failure steps
        self.breaking_point_found = False
        self.breaking_point_users = None
        
    def run_concurrent_requests(self, num_users: int, step: int) -> List[RequestResult]:
        """Run multiple TRULY CONCURRENT requests using threading barrier"""
        results = []
        
        if RICH_AVAILABLE:
            console.print(f"[dim]Launching {num_users} threads simultaneously...[/dim]")
        else:
            print(f"Launching {num_users} threads simultaneously...")
        
        # Create a barrier to synchronize thread start - ensures true concurrency
        start_barrier = threading.Barrier(num_users)
        
        # Track start time for actual concurrent execution
        actual_start_time = None
        start_lock = threading.Lock()
        
        def set_start_time():
            nonlocal actual_start_time
            with start_lock:
                if actual_start_time is None:
                    actual_start_time = time.time()
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_users) as executor:
            futures = []
            questions = [random.choice(QUESTIONS) for _ in range(num_users)]
            
            # Submit all requests at once
            for user_idx in range(num_users):
                future = executor.submit(
                    self._send_request_wrapper,
                    questions[user_idx],
                    step,
                    user_idx,
                    start_barrier,
                    set_start_time
                )
                futures.append(future)
            
            if RICH_AVAILABLE:
                console.print(f"[dim]All threads ready, firing requests...[/dim]")
            else:
                print(f"All threads ready, firing requests...")
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    # Show progress
                    if completed % 5 == 0 or completed == num_users:
                        status_icon = "✓" if result.success else "✗"
                        if RICH_AVAILABLE:
                            color = "green" if result.success else "red"
                            console.print(f"[dim]Progress: {completed}/{num_users} [{color}]{status_icon}[/{color}] User {result.user_index}: {result.status} ({result.latency_ms:.0f}ms)[/dim]")
                        else:
                            print(f"Progress: {completed}/{num_users} {status_icon} User {result.user_index}: {result.status} ({result.latency_ms:.0f}ms)")
                    
                except Exception as e:
                    if RICH_AVAILABLE:
                        console.print(f"[red]Error collecting result: {e}[/red]")
                    else:
                        print(f"Error collecting result: {e}")
        
        # Check queue status after the burst
        time.sleep(0.5)  # Brief wait for queue to settle
        queue_status = self.client.get_queue_status()
        if queue_status:
            if RICH_AVAILABLE:
                console.print(f"[cyan]Queue after burst: Active={queue_status.get('active_requests', 'N/A')}, "
                             f"Queued={queue_status.get('queued_requests', 'N/A')}[/cyan]")
            else:
                print(f"Queue after burst: Active={queue_status.get('active_requests', 'N/A')}, "
                      f"Queued={queue_status.get('queued_requests', 'N/A')}")
        
        return results
    
    def _send_request_wrapper(self, question: str, step: int, user_idx: int, 
                              start_barrier: threading.Barrier, set_start_time_func):
        """Wrapper to track actual start time"""
        # Wait for all threads to be ready
        start_barrier.wait()
        set_start_time_func()
        
        # Now all threads fire at once!
        return self.client.send_request(question, step, user_idx, start_barrier)
    
    def calculate_step_metrics(self, results: List[RequestResult], step: int, duration: float) -> StepMetrics:
        """Calculate aggregated metrics for a step"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        success_rate = (successful / total * 100) if total > 0 else 0
        fail_rate = 100 - success_rate
        
        # Latency statistics (only for successful requests)
        latencies = [r.latency_ms for r in results if r.latency_ms is not None]
        
        if latencies:
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
            p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0]
        else:
            avg_latency = median_latency = min_latency = max_latency = p95_latency = p99_latency = 0
        
        # Error breakdown - categorize by status and error message
        errors = {}
        for r in results:
            if not r.success:
                # Categorize by status first
                error_key = f"[{r.status}] {r.error_message or 'Unknown error'}"
                errors[error_key] = errors.get(error_key, 0) + 1
        
        # Throughput
        throughput = total / duration if duration > 0 else 0
        
        # DEBUGGING: Calculate concurrency metrics
        barrier_wait_times = [r.barrier_wait_time_ms for r in results if r.barrier_wait_time_ms is not None]
        avg_barrier_wait = statistics.mean(barrier_wait_times) if barrier_wait_times else 0
        
        # Count how many requests actually started within 100ms of each other (true concurrency)
        start_times = [r.request_start_time for r in results if r.request_start_time is not None]
        if start_times:
            min_start = min(start_times)
            actual_concurrent_starts = sum(1 for t in start_times if t - min_start <= 0.1)  # Within 100ms
        else:
            actual_concurrent_starts = 0
        
        return StepMetrics(
            step=step,
            concurrent_users=len(set(r.user_index for r in results)),
            total_requests=total,
            successful_requests=successful,
            failed_requests=failed,
            success_rate=success_rate,
            fail_rate=fail_rate,
            avg_latency_ms=avg_latency,
            median_latency_ms=median_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            throughput_rps=throughput,
            duration_seconds=duration,
            errors=errors,
            barrier_wait_time_ms=avg_barrier_wait,
            actual_concurrent_starts=actual_concurrent_starts
        )
    
    def run_test(self):
        """Run the complete incremental load test to find breaking point"""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]🚀 Starting Breaking Point Detection Test[/bold cyan]\n")
            console.print(f"[cyan]Configuration:[/cyan]")
            console.print(f"  • Users: {self.config.MIN_USERS} → {self.config.MAX_USERS} (increment: {self.config.USERS_INCREMENT})")
            console.print(f"  • API: {self.config.API_URL}")
            console.print(f"  • Failure Threshold: {self.config.FAILURE_THRESHOLD}%")
            console.print(f"  • Stop after {self.config.CONSECUTIVE_FAILURES} consecutive high-failure steps\n")
        else:
            print("\n🚀 Starting Breaking Point Detection Test\n")
            print(f"Users: {self.config.MIN_USERS} → {self.config.MAX_USERS}")
            print(f"API: {self.config.API_URL}")
            print(f"Failure Threshold: {self.config.FAILURE_THRESHOLD}%\n")
        
        # Check initial queue status
        queue_status = self.client.get_queue_status()
        if queue_status and RICH_AVAILABLE:
            console.print(f"[green]✓ API Queue Status:[/green] Active: {queue_status.get('active_requests', 'N/A')}, "
                         f"Queued: {queue_status.get('queued_requests', 'N/A')}, "
                         f"Max: {queue_status.get('max_concurrent', 'N/A')}\n")
        
        step = 0
        try:
            for num_users in range(self.config.MIN_USERS, self.config.MAX_USERS + 1, self.config.USERS_INCREMENT):
                step += 1
                
                if RICH_AVAILABLE:
                    console.print(f"\n[bold yellow]📊 Step {step}: Testing with {num_users} concurrent users[/bold yellow]")
                else:
                    print(f"\n📊 Step {step}: Testing with {num_users} concurrent users")
                
                # Check queue before firing requests
                queue_before = self.client.get_queue_status()
                if queue_before:
                    if RICH_AVAILABLE:
                        console.print(f"[dim]Queue before: Active={queue_before.get('active_requests', 0)}, "
                                    f"Queued={queue_before.get('queued_requests', 0)}[/dim]")
                    else:
                        print(f"Queue before: Active={queue_before.get('active_requests', 0)}, "
                              f"Queued={queue_before.get('queued_requests', 0)}")
                
                step_start = time.time()
                
                # Run concurrent requests
                step_results = self.run_concurrent_requests(num_users, step)
                
                step_duration = time.time() - step_start
                
                # Calculate metrics
                metrics = self.calculate_step_metrics(step_results, step, step_duration)
                
                # Store results
                self.all_results.extend(step_results)
                self.step_metrics.append(metrics)
                
                # Display step results
                self._display_step_results(metrics)
                
                # Check for breaking point
                stop_reason = self._check_breaking_point(metrics)
                if stop_reason:
                    if RICH_AVAILABLE:
                        console.print(f"\n[bold red]🛑 {stop_reason}[/bold red]")
                        console.print(f"[bold yellow]Breaking point: {metrics.concurrent_users} concurrent users[/bold yellow]")
                    else:
                        print(f"\n🛑 {stop_reason}")
                        print(f"Breaking point: {metrics.concurrent_users} concurrent users")
                    self.breaking_point_found = True
                    self.breaking_point_users = metrics.concurrent_users
                    break
                
                # Wait before next step
                if num_users < self.config.MAX_USERS:
                    time.sleep(self.config.WAIT_BETWEEN_STEPS)
        
        except KeyboardInterrupt:
            if RICH_AVAILABLE:
                console.print("\n[yellow]⚠️  Test interrupted by user[/yellow]")
            else:
                print("\n⚠️  Test interrupted by user")
        
        finally:
            self.client.close()
        
        # Save results
        self._save_results()
        
        # Display final summary
        self._display_summary()
    
    def _display_step_results(self, metrics: StepMetrics):
        """Display results for a single step"""
        if RICH_AVAILABLE:
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            
            table.add_row("Total Requests", str(metrics.total_requests))
            table.add_row("Successful", f"[green]{metrics.successful_requests}[/green]")
            table.add_row("Failed", f"[red]{metrics.failed_requests}[/red]")
            table.add_row("Success Rate", f"{metrics.success_rate:.1f}%")
            table.add_row("Fail Rate", f"[yellow]{metrics.fail_rate:.1f}%[/yellow]")
            table.add_row("Avg Latency", f"{metrics.avg_latency_ms:.0f} ms")
            table.add_row("P95 Latency", f"{metrics.p95_latency_ms:.0f} ms")
            table.add_row("Throughput", f"{metrics.throughput_rps:.2f} req/s")
            
            # DEBUGGING: Show concurrency verification
            if hasattr(metrics, 'barrier_wait_time_ms') and metrics.barrier_wait_time_ms > 0:
                table.add_row("Barrier Wait Time", f"{metrics.barrier_wait_time_ms:.1f} ms")
            if hasattr(metrics, 'actual_concurrent_starts') and metrics.actual_concurrent_starts > 0:
                concurrent_pct = (metrics.actual_concurrent_starts / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
                table.add_row("True Concurrent Starts", f"{metrics.actual_concurrent_starts}/{metrics.total_requests} ({concurrent_pct:.1f}%)")
            
            # Show error breakdown if there are errors
            if metrics.errors:
                table.add_row("", "")
                table.add_row("Error Breakdown", "[bold red]Details:[/bold red]")
                for error, count in list(metrics.errors.items())[:5]:  # Show top 5 errors
                    table.add_row("", f"[red]• {error[:60]}: {count}x[/red]")
            
            console.print(table)
        else:
            print(f"  Total: {metrics.total_requests} | Success: {metrics.successful_requests} | "
                  f"Failed: {metrics.failed_requests} | Success Rate: {metrics.success_rate:.1f}% | "
                  f"Fail Rate: {metrics.fail_rate:.1f}%")
            print(f"  Avg Latency: {metrics.avg_latency_ms:.0f}ms | P95: {metrics.p95_latency_ms:.0f}ms | "
                  f"Throughput: {metrics.throughput_rps:.2f} req/s")
            
            # DEBUGGING: Show concurrency verification
            if hasattr(metrics, 'barrier_wait_time_ms') and metrics.barrier_wait_time_ms > 0:
                print(f"  Barrier Wait Time: {metrics.barrier_wait_time_ms:.1f}ms")
            if hasattr(metrics, 'actual_concurrent_starts') and metrics.actual_concurrent_starts > 0:
                concurrent_pct = (metrics.actual_concurrent_starts / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
                print(f"  True Concurrent Starts: {metrics.actual_concurrent_starts}/{metrics.total_requests} ({concurrent_pct:.1f}%)")
            
            # Show errors
            if metrics.errors:
                print(f"  Errors:")
                for error, count in list(metrics.errors.items())[:5]:
                    print(f"    • {error[:60]}: {count}x")
    
    def _check_breaking_point(self, metrics: StepMetrics) -> Optional[str]:
        """Check if we've reached the API breaking point"""
        
        # Check if failure rate exceeds threshold
        if metrics.fail_rate >= self.config.FAILURE_THRESHOLD:
            self.consecutive_high_failures += 1
            
            if self.consecutive_high_failures >= self.config.CONSECUTIVE_FAILURES:
                return f"API Breaking Point Detected: {self.config.CONSECUTIVE_FAILURES} consecutive steps with >{self.config.FAILURE_THRESHOLD}% failure rate"
        else:
            # Reset counter if we get a good step
            self.consecutive_high_failures = 0
        
        # Check for complete API death (all connection errors or timeouts)
        connection_errors = sum(1 for r in self.all_results[-metrics.total_requests:] 
                               if r.status in ["connection_error", "timeout"])
        if connection_errors >= metrics.total_requests * 0.8:  # 80% connection/timeout errors
            return f"API Appears Dead: {connection_errors}/{metrics.total_requests} requests failed with connection/timeout errors"
        
        # Check for catastrophic failure rate
        if metrics.fail_rate >= 80:
            return f"Catastrophic Failure: {metrics.fail_rate:.1f}% failure rate"
        
        return None
    
    def _should_stop_test(self, metrics: StepMetrics) -> bool:
        """Check if test should stop based on metrics"""
        return metrics.fail_rate > self.config.MAX_ACCEPTABLE_FAIL_RATE
    
    def _display_summary(self):
        """Display final test summary"""
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]📈 Test Summary[/bold cyan]\n")
            
            # Show breaking point if found
            if self.breaking_point_found:
                console.print(f"[bold red]🎯 API Breaking Point: {self.breaking_point_users} concurrent users[/bold red]\n")
            else:
                console.print(f"[bold green]✓ API handled up to {self.step_metrics[-1].concurrent_users} concurrent users without breaking[/bold green]\n")
            
            table = Table(title="Performance Metrics by Load Step")
            table.add_column("Step", style="cyan", justify="right")
            table.add_column("Users", justify="right")
            table.add_column("Total", justify="right")
            table.add_column("Success", justify="right", style="green")
            table.add_column("Failed", justify="right", style="red")
            table.add_column("Success %", justify="right")
            table.add_column("Fail %", justify="right")
            table.add_column("Avg Latency", justify="right")
            table.add_column("P95 Latency", justify="right")
            table.add_column("Throughput", justify="right")
            
            for m in self.step_metrics:
                success_style = "green" if m.success_rate > 90 else ("yellow" if m.success_rate > 70 else "red")
                table.add_row(
                    str(m.step),
                    str(m.concurrent_users),
                    str(m.total_requests),
                    str(m.successful_requests),
                    str(m.failed_requests),
                    f"[{success_style}]{m.success_rate:.1f}%[/{success_style}]",
                    f"{m.fail_rate:.1f}%",
                    f"{m.avg_latency_ms:.0f}ms",
                    f"{m.p95_latency_ms:.0f}ms",
                    f"{m.throughput_rps:.2f}",
                )
            
            console.print(table)
            
            # Show status distribution
            console.print("\n[bold cyan]📊 Overall Status Distribution[/bold cyan]")
            status_counts = {}
            for r in self.all_results:
                status_counts[r.status] = status_counts.get(r.status, 0) + 1
            
            status_table = Table(show_header=True)
            status_table.add_column("Status", style="cyan")
            status_table.add_column("Count", justify="right")
            status_table.add_column("Percentage", justify="right")
            
            total_results = len(self.all_results)
            for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_results * 100) if total_results > 0 else 0
                color = "green" if status == "success" else "red"
                status_table.add_row(
                    f"[{color}]{status}[/{color}]",
                    str(count),
                    f"{pct:.1f}%"
                )
            
            console.print(status_table)
            
        else:
            print("\n📈 Test Summary\n")
            
            # Show breaking point if found
            if self.breaking_point_found:
                print(f"🎯 API Breaking Point: {self.breaking_point_users} concurrent users\n")
            else:
                print(f"✓ API handled up to {self.step_metrics[-1].concurrent_users} concurrent users without breaking\n")
            
            for m in self.step_metrics:
                print(f"Step {m.step} ({m.concurrent_users} users): "
                      f"{m.successful_requests}/{m.total_requests} success ({m.success_rate:.1f}%), "
                      f"fail rate: {m.fail_rate:.1f}%, "
                      f"Avg: {m.avg_latency_ms:.0f}ms, P95: {m.p95_latency_ms:.0f}ms")
            
            # Status distribution
            print("\n📊 Overall Status Distribution:")
            status_counts = {}
            for r in self.all_results:
                status_counts[r.status] = status_counts.get(r.status, 0) + 1
            
            total_results = len(self.all_results)
            for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total_results * 100) if total_results > 0 else 0
                print(f"  {status}: {count} ({pct:.1f}%)")
    
    def _save_results(self):
        """Save results to Excel with multiple sheets and formatting"""
        if RICH_AVAILABLE:
            console.print("\n[cyan]💾 Saving results...[/cyan]")
        else:
            print("\n💾 Saving results...")
        
        # Create output directory
        self.config.OUTPUT_DIR.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        breaking_point_suffix = f"_bp{self.breaking_point_users}" if self.breaking_point_found else "_no_bp"
        filename = self.config.OUTPUT_DIR / f"graphrag_performance_test_{timestamp}{breaking_point_suffix}.xlsx"
        
        try:
            with pd.ExcelWriter(filename, engine="openpyxl") as writer:
                # Summary sheet
                summary_df = pd.DataFrame([m.to_dict() for m in self.step_metrics])
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Detailed results sheet
                detailed_df = pd.DataFrame([r.to_dict() for r in self.all_results])
                detailed_df.to_excel(writer, sheet_name="Detailed_Results", index=False)
                
                # Results by step
                for metrics in self.step_metrics:
                    step_results = [r for r in self.all_results if r.step == metrics.step]
                    if step_results:
                        step_df = pd.DataFrame([r.to_dict() for r in step_results])
                        sheet_name = f"Step_{metrics.step}_{metrics.concurrent_users}users"
                        step_df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit
                
                # Statistics sheet
                if self.all_results:
                    stats_data = self._generate_statistics()
                    stats_df = pd.DataFrame([stats_data])
                    stats_df.to_excel(writer, sheet_name="Statistics", index=False)
            
            if RICH_AVAILABLE:
                console.print(f"[green]✓ Results saved to: {filename}[/green]")
            else:
                print(f"✓ Results saved to: {filename}")
                
        except Exception as e:
            if RICH_AVAILABLE:
                console.print(f"[red]✗ Error saving results: {e}[/red]")
            else:
                print(f"✗ Error saving results: {e}")
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """Generate overall test statistics"""
        total_requests = len(self.all_results)
        successful = sum(1 for r in self.all_results if r.success)
        
        latencies = [r.latency_ms for r in self.all_results if r.latency_ms is not None]
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": total_requests - successful,
            "overall_success_rate": (successful / total_requests * 100) if total_requests > 0 else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0,
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else (latencies[0] if latencies else 0),
            "p99_latency_ms": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else (latencies[0] if latencies else 0),
            "total_steps": len(self.step_metrics),
            "max_concurrent_users_tested": max((m.concurrent_users for m in self.step_metrics), default=0),
            "breaking_point_found": self.breaking_point_found,
            "breaking_point_users": self.breaking_point_users if self.breaking_point_found else "N/A",
        }


# ========================================
# MAIN
# ========================================

def main():
    """Main entry point"""
    config = TestConfig()
    
    # Create output directory
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Run test
    executor = LoadTestExecutor(config)
    executor.run_test()
    
    if RICH_AVAILABLE:
        console.print("\n[bold green]✓ Test completed![/bold green]\n")
    else:
        print("\n✓ Test completed!\n")


if __name__ == "__main__":
    main()
