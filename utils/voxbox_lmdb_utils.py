import lmdb
from RawMutltipleWebDataset import RawMutltipleWebDataset
import glob
import tqdm
import click
import json
import os
import hashlib
def open_lmdb_for_read(lmdb_path):
    env = lmdb.open(lmdb_path, map_size=1024*1024*1024*10, readonly=True)
    return env

def get_json_from_lmdb(lmdb_env, text):
    with lmdb_env.begin(write=False) as txn:
        json_data = txn.get(text.encode())
    return json_data

def create_lmdb(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    data_files = glob.glob(os.path.join(input_dir, "*.tar"))
    print(f"Found {len(data_files)} data files")
    lmdb_path = os.path.join(output_dir, "voxbox.lmdb")
    env = lmdb.open(lmdb_path, map_size=1024*1024*1024*10)
    with env.begin(write=True) as txn:
        for data_file in tqdm.tqdm(data_files):
            dataset = RawMutltipleWebDataset([data_file])
            dataset_progress = tqdm.tqdm(dataset, desc=f"Processing {data_file}")
            for item in dataset_progress:
                try:
                    json_data = json.loads(item['json'])
                    text = json_data['text']
                    if text is None or text == "":
                        print(f"Text is None for {item['json']}")
                        continue
                    object = json_data
                    object.pop('text')
                    object.pop('syllables')
                    key = hashlib.md5(text.encode()).hexdigest()
                    txn.put(key.encode(), json.dumps(object,ensure_ascii=False).encode())
                except Exception as e:
                    print(f"Error processing {item['json']}: {e}")
                    raise e
    env.close()
    print(f"Voxbox LMDB created at {lmdb_path}")

@click.command()
@click.option("--task", type=str, required=True)
@click.option("--input_dir", type=str, required=False)
@click.option("--lmdb_path", type=str, required=False)
@click.option("--output_dir", type=str, required=False, default=None)
@click.option("--text", type=str, required=False, default=None)
def main(task, input_dir, lmdb_path, output_dir, text):
    if task == "create_lmdb":
        create_lmdb(input_dir, output_dir)
    elif task == "get_json":
        env = open_lmdb_for_read(lmdb_path)
        result = get_json_from_lmdb(env, text)
        print(result)
    elif task == "create_multiple_lmdb":
        #list the subfolders in the input_dir
        subfolders = os.listdir(input_dir)
        for subfolder in subfolders:
            print(f"Creating LMDB for {subfolder}")
            create_lmdb(os.path.join(input_dir, subfolder), os.path.join(output_dir, subfolder))
            print(f"LMDB for {subfolder} created")







if __name__ == "__main__":
    main()