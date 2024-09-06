import json
from concurrent.futures import ThreadPoolExecutor
import argparse
from benchmark.datasets import DATASETS
import gc

nb = 1000000000
# Function to handle a thread's work and save results into a JSON file
def process_and_save_to_json(processor, thread_no, total_threads):
    print(f"Starting thread {thread_no}")
    try:
        split = (total_threads, thread_no)
        offset = thread_no * (nb // total_threads)
        print(f"Starting offset {offset}")

        # Open file in append mode to save chunks of data iteratively
        filename = f'./json_data/thread_{thread_no}.json'
        with open(filename, 'w') as f:
            f.write('[')  # Start of JSON array

        first_entry = True
        for batch in processor.get_dataset_iterator(split=split):
            data = []
            for row in batch:
                d = {"id": offset, "emb": row.tolist()}
                data.append(d)
                offset += 1  # Increment offset for each row

            # Write data chunk to file in JSON format
            with open(filename, 'a') as f:
                if first_entry:
                    json.dump(data, f)
                    first_entry = False
                else:
                    f.write(', ')
                    json.dump(data, f)

            # Free memory from the processed batch
            del data, batch
            gc.collect()  # Force garbage collection to free memory

        # End of JSON array
        with open(filename, 'a') as f:
            f.write(']')

        print(f"Thread {thread_no} processed and saved data to {filename}.")
    except Exception as e:
        print(f"Error processing data in thread {thread_no}: {str(e)}")
# Main function to run multi-threading
def run_multi_threaded(processor, total_threads=100):
    # Create the dataset processor

    # Create a thread pool with `total_threads` threads
    with ThreadPoolExecutor(max_workers=100) as executor:
        for thread_no in range(total_threads):
            executor.submit(process_and_save_to_json, processor, thread_no, total_threads)

# Example call to run the function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    parser.add_argument(
        '--skip-data',
        action='store_true',
        help='skip downloading base vectors')
    args = parser.parse_args()
    ds = DATASETS[args.dataset]()
    ds.prepare(True if args.skip_data else False)
    
    run_multi_threaded(ds)
