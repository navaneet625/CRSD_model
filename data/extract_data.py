import os

def extract_first_n_megabytes(input_filename: str, output_filename: str, size_mb: int):
    """
    Reads the first N megabytes from the input file and writes them to the output file.
    """
    
    # Calculate the required size in bytes (1 MB = 1024 * 1024 bytes)
    BYTES_PER_MB = 1024 * 1024
    target_bytes = size_mb * BYTES_PER_MB
    
    try:
        # 1. Open the input file in binary read mode ('rb')
        with open(input_filename, 'rb') as infile:
            
            # 2. Read the exact number of bytes
            data_chunk = infile.read(target_bytes)
            
            # Check if the file was smaller than the target size
            actual_size = len(data_chunk)
            
            # 3. Open the output file in binary write mode ('wb')
            with open(output_filename, 'wb') as outfile:
                outfile.write(data_chunk)

        print(f"✅ Successfully read {actual_size / BYTES_PER_MB:.2f} MB from '{input_filename}'.")
        print(f"💾 Saved data to '{output_filename}'.")

    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_filename}' not found.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

# --- Configuration ---
INPUT_FILE = "enwik8"
OUTPUT_FILE = "enwik8_2M.txt"
TARGET_MB = 2

# Run the extraction
extract_first_n_megabytes(INPUT_FILE, OUTPUT_FILE, TARGET_MB)

if __name__ == "__main__":
    extract_first_n_megabytes("data/enwik8", "data/enwik8_2M.txt", 2)


