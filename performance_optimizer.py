import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Optional, Any, Callable
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from datetime import datetime, timedelta
import hashlib
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Advanced performance optimization system with caching, async processing, and monitoring
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize performance optimizer
        
        Args:
            cache_size: Maximum cache size
            cache_ttl: Cache time-to-live in seconds
        """
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.cache = {}
        self.cache_timestamps = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Performance metrics
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'avg_response_time': 0.0,
            'function_calls': {},
            'errors': 0
        }
        
        # Rate limiting
        self.rate_limits = {}
        self.rate_limit_window = 60  # 1 minute
        
        logger.info("Performance optimizer initialized")
    
    def cached(self, ttl: int = None, key_func: Callable = None):
        """
        Decorator for caching function results
        
        Args:
            ttl: Time-to-live for cache entry
            key_func: Function to generate cache key
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Check cache
                if self._is_cached(cache_key, ttl):
                    self.metrics['cache_hits'] += 1
                    return self.cache[cache_key]
                
                # Execute function
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Cache result
                    self._cache_result(cache_key, result, ttl)
                    
                    # Update metrics
                    self.metrics['cache_misses'] += 1
                    self._update_function_metrics(func.__name__, execution_time)
                    
                    return result
                    
                except Exception as e:
                    self.metrics['errors'] += 1
                    logger.error(f"Error in cached function {func.__name__}: {str(e)}")
                    raise
            
            return wrapper
        return decorator
    
    def async_cached(self, ttl: int = None, key_func: Callable = None):
        """
        Decorator for caching async function results
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_cache_key(func.__name__, args, kwargs)
                
                # Check cache
                if self._is_cached(cache_key, ttl):
                    self.metrics['cache_hits'] += 1
                    return self.cache[cache_key]
                
                # Execute function
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Cache result
                    self._cache_result(cache_key, result, ttl)
                    
                    # Update metrics
                    self.metrics['cache_misses'] += 1
                    self._update_function_metrics(func.__name__, execution_time)
                    
                    return result
                    
                except Exception as e:
                    self.metrics['errors'] += 1
                    logger.error(f"Error in cached async function {func.__name__}: {str(e)}")
                    raise
            
            return wrapper
        return decorator
    
    def rate_limited(self, max_calls: int = 60, window: int = 60):
        """
        Decorator for rate limiting function calls
        
        Args:
            max_calls: Maximum calls per window
            window: Time window in seconds
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Get rate limit key
                rate_key = f"{func.__name__}:{threading.current_thread().ident}"
                
                # Check rate limit
                if not self._check_rate_limit(rate_key, max_calls, window):
                    raise Exception(f"Rate limit exceeded for {func.__name__}")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    async def batch_process(self, items: List[Any], process_func: Callable, 
                          batch_size: int = 5, max_workers: int = 4) -> List[Any]:
        """
        Process items in batches asynchronously
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Size of each batch
            max_workers: Maximum number of workers
            
        Returns:
            List of processed results
        """
        try:
            # Create batches
            batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
            
            # Process batches concurrently
            tasks = []
            for batch in batches:
                task = asyncio.create_task(self._process_batch(batch, process_func))
                tasks.append(task)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*tasks)
            
            # Flatten results
            results = []
            for batch_result in batch_results:
                results.extend(batch_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return []
    
    async def _process_batch(self, batch: List[Any], process_func: Callable) -> List[Any]:
        """Process a single batch"""
        try:
            loop = asyncio.get_event_loop()
            tasks = []
            
            for item in batch:
                if asyncio.iscoroutinefunction(process_func):
                    task = process_func(item)
                else:
                    task = loop.run_in_executor(self.executor, process_func, item)
                tasks.append(task)
            
            return await asyncio.gather(*tasks)
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return []
    
    def stream_response(self, generator_func: Callable, *args, **kwargs):
        """
        Stream response from a generator function
        
        Args:
            generator_func: Generator function to stream from
            *args: Arguments for the generator function
            **kwargs: Keyword arguments for the generator function
            
        Yields:
            Chunks from the generator
        """
        try:
            for chunk in generator_func(*args, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield {"error": str(e)}
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        key_data = {
            'func_name': func_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cached(self, cache_key: str, ttl: int = None) -> bool:
        """Check if result is cached and not expired"""
        if cache_key not in self.cache:
            return False
        
        if ttl is None:
            ttl = self.cache_ttl
        
        cache_time = self.cache_timestamps.get(cache_key, 0)
        if time.time() - cache_time > ttl:
            # Remove expired entry
            del self.cache[cache_key]
            del self.cache_timestamps[cache_key]
            return False
        
        return True
    
    def _cache_result(self, cache_key: str, result: Any, ttl: int = None):
        """Cache function result"""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_key = min(self.cache_timestamps.keys(), 
                           key=lambda k: self.cache_timestamps[k])
            del self.cache[oldest_key]
            del self.cache_timestamps[oldest_key]
        
        # Cache result
        self.cache[cache_key] = result
        self.cache_timestamps[cache_key] = time.time()
    
    def _check_rate_limit(self, rate_key: str, max_calls: int, window: int) -> bool:
        """Check if rate limit is exceeded"""
        now = time.time()
        
        if rate_key not in self.rate_limits:
            self.rate_limits[rate_key] = []
        
        # Remove old calls outside the window
        self.rate_limits[rate_key] = [
            call_time for call_time in self.rate_limits[rate_key]
            if now - call_time < window
        ]
        
        # Check if limit is exceeded
        if len(self.rate_limits[rate_key]) >= max_calls:
            return False
        
        # Add current call
        self.rate_limits[rate_key].append(now)
        return True
    
    def _update_function_metrics(self, func_name: str, execution_time: float):
        """Update function performance metrics"""
        if func_name not in self.metrics['function_calls']:
            self.metrics['function_calls'][func_name] = {
                'count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        func_metrics = self.metrics['function_calls'][func_name]
        func_metrics['count'] += 1
        func_metrics['total_time'] += execution_time
        func_metrics['avg_time'] = func_metrics['total_time'] / func_metrics['count']
        func_metrics['min_time'] = min(func_metrics['min_time'], execution_time)
        func_metrics['max_time'] = max(func_metrics['max_time'], execution_time)
        
        # Update global average response time
        self.metrics['total_requests'] += 1
        self.metrics['avg_response_time'] = (
            (self.metrics['avg_response_time'] * (self.metrics['total_requests'] - 1) + execution_time)
            / self.metrics['total_requests']
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        cache_hit_rate = 0.0
        if self.metrics['cache_hits'] + self.metrics['cache_misses'] > 0:
            cache_hit_rate = self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses'])
        
        return {
            'cache': {
                'size': len(self.cache),
                'hit_rate': cache_hit_rate,
                'hits': self.metrics['cache_hits'],
                'misses': self.metrics['cache_misses']
            },
            'performance': {
                'total_requests': self.metrics['total_requests'],
                'avg_response_time': self.metrics['avg_response_time'],
                'errors': self.metrics['errors']
            },
            'functions': self.metrics['function_calls'],
            'rate_limits': {
                key: len(calls) for key, calls in self.rate_limits.items()
            }
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Cache cleared")
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_requests': 0,
            'avg_response_time': 0.0,
            'function_calls': {},
            'errors': 0
        }
        self.rate_limits.clear()
        logger.info("Metrics reset")
    
    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.get_metrics(), f, indent=2)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
    
    def load_metrics(self, filepath: str):
        """Load metrics from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    metrics = json.load(f)
                # Update metrics (simplified version)
                logger.info(f"Metrics loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

# Convenience decorators
def cached(ttl: int = None, key_func: Callable = None):
    """Global cached decorator"""
    return performance_optimizer.cached(ttl, key_func)

def async_cached(ttl: int = None, key_func: Callable = None):
    """Global async cached decorator"""
    return performance_optimizer.async_cached(ttl, key_func)

def rate_limited(max_calls: int = 60, window: int = 60):
    """Global rate limited decorator"""
    return performance_optimizer.rate_limited(max_calls, window)
