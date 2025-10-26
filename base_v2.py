import asyncio
from openai import AsyncOpenAI
import time


async def make_request(client, request_id):
    start = time.time()
    response = await client.chat.completions.create(
        model="gemma-3-1b-it",
        messages=[
            {"role": "user",
                "content": f"What is a butterfly? (Request {request_id})"}
        ],
    )
    elapsed = time.time() - start
    print(f"Request {request_id} completed in {elapsed:.2f}s")
    return response.choices[0].message.content


async def test_concurrent_requests(num_requests=5):
    client = AsyncOpenAI(
        base_url="http://localhost:5050/v1",
        api_key="not-needed"
    )

    start_time = time.time()

    # Create multiple concurrent requests
    tasks = [make_request(client, i) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    print(
        f"\nTotal time for {num_requests} concurrent requests: {total_time:.2f}s")
    print(f"Average time per request: {total_time/num_requests:.2f}s")

    return results

# Run the test
asyncio.run(test_concurrent_requests(5))
