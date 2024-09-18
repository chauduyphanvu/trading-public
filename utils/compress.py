import bz2
import gzip
import json
import lzma
import time
import zlib

import brotli
import lz4.frame as lz4f
import snappy
import zstandard as zstd


# Function to read a JSON file
def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Function to write compressed file and measure compression time
def compress_and_measure(data, compress_func, file_ext):
    start_time = time.time()
    compressed_data = compress_func(json.dumps(data).encode("utf-8"))
    end_time = time.time()
    compress_time = end_time - start_time
    compressed_size = len(compressed_data)

    output_file = f"compressed_file.{file_ext}"
    with open(output_file, "wb") as file:
        file.write(compressed_data)

    return compressed_size, compress_time, output_file


# Compression functions
def gzip_compress(data):
    return gzip.compress(data)


def bz2_compress(data):
    return bz2.compress(data)


def lzma_compress(data):
    return lzma.compress(data)


def brotli_compress(data):
    return brotli.compress(data)


def zstd_compress(data):
    return zstd.compress(data)


def lz4_compress(data):
    return lz4f.compress(data)


def snappy_compress(data):
    return snappy.compress(data)


def zlib_compress(data):
    return zlib.compress(data)


# Main function to compress using various methods and compare them
def main(json_file_path):
    data = read_json(json_file_path)

    compression_methods = {
        "gzip": gzip_compress,
        "bz2": bz2_compress,
        "lzma": lzma_compress,
        "brotli": brotli_compress,
        "zstd": zstd_compress,
        "lz4": lz4_compress,
        "snappy": snappy_compress,
        "zlib": zlib_compress,
    }

    results = []
    for method, func in compression_methods.items():
        size, time_taken, output_file = compress_and_measure(data, func, method)
        results.append(
            {
                "method": method,
                "compressed_size": size,
                "compression_time": time_taken,
                "output_file": output_file,
            }
        )

    # Print the comparison results
    print(f"{'Method':<10} {'Size (bytes)':<15} {'Time (s)':<10} {'Output File'}")
    for result in results:
        print(
            f"{result['method']:<10} {result['compressed_size']:<15} {result['compression_time']:<10.6f} {result['output_file']}"
        )


if __name__ == "__main__":
    json_file_path = "../data/generated/ta-trading-signals.json"
    main(json_file_path)
