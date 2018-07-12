"""
Copyright 2018 Nadheesh Jihan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Use this directory to

import zipfile
from os import listdir
from os.path import join, exists

import pandas as pd

# PARAMETERS
window_size = 10000  # window used to compute the avg latency and throughput
root_dir = "1G_heap/https_transformation.bal_bal/default_flags"  # root directory


def get_column_value(dir_name, level):
    value = dir_name.split("_")[0]

    if level == "users":
        return int(value)
    elif level == "m_size":
        return int(value[:-1])
    elif level == "sleep":
        return int(value[:-2])

    return None


def extract_data(window_size, root_dir):
    """
    Extracts data from .jtls
    :param window_size:
    :param root_dir:
    :param out_path:
    :return:
    """
    n_users = listdir(root_dir)
    for n_user in n_users:
        user_dir = join(root_dir, n_user)
        n_user = get_column_value(n_user, "users")

        m_sizes = listdir(user_dir)
        for m_size in m_sizes:
            m_dir = join(user_dir, m_size)
            m_size = get_column_value(m_size, "m_size")

            sleeps = listdir(m_dir)
            for sleep in sleeps:
                sleep_dir = join(m_dir, sleep)
                sleep = get_column_value(sleep, "sleep")

                jtl_root = join(sleep_dir, "jtls")
                jtl_file = "results-measurement.jtl"
                if not exists(join(jtl_root, jtl_file)):
                    zip_ref = zipfile.ZipFile(join(sleep_dir, "jtls.zip"), 'r')
                    zip_ref.extract(jtl_file, jtl_root)
                    zip_ref.close()
                print(n_user, m_size, sleep)
                df = pd.read_csv(join(jtl_root, jtl_file))
                print(df.shape)

                avg_latency = []
                throughput = []
                elapsed = df.elapsed.values
                timestamps = df.timeStamp.values

                df = None

                for i in range(0, elapsed.shape[0], window_size):
                    elapsed_bin = elapsed[i:i + window_size]
                    timestamps_bin = timestamps[i:i + window_size]
                    avg_latency.append(elapsed_bin.mean(axis=0))
                    throughput.append((timestamps_bin.shape[0] / (timestamps_bin[-1] - timestamps_bin[0])) * 1000)

                size = len(avg_latency)
                df = pd.DataFrame({
                    "users": [n_user] * size,
                    "message_size": [m_size] * size,
                    "sleep_time": [sleep] * size,
                    "throughput": throughput,
                    "latency": avg_latency
                })

                out_file = "data_with_win_%i.csv" % window_size
                if exists(out_file):
                    with open(out_file, 'a') as f:
                        df.to_csv(f, header=False)
                else:
                    df.to_csv(out_file)


if __name__ == '__main__':
    extract_data(window_size, root_dir)
