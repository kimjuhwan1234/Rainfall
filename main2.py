import os
import time
import torch
import pandas as pd
from Module.run import Run
from utils.parser import config
from sklearn.model_selection import train_test_split
from multiprocessing import set_start_method, Process


class Execution:
    def __init__(self, directory, saving_path):
        self.config = config
        self.directory = directory
        self.saving_path = saving_path

    def setup(self, rank):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        torch.cuda.set_device(rank)
        print(f"Process on GPU: {torch.cuda.current_device()}")
        # logging.info(f"Process on GPU: {torch.cuda.current_device()}")

    def process_file(self, rank, X_train, y_train, X_val, y_val, weight_path):
        self.setup(rank)

        trainer = Run(X_train, y_train, X_val, y_val, weight_path, self.config)
        trainer.load_data()
        trainer.run_model()
        trainer.check_validation()

    # def main(self):
    #     set_start_method('spawn', force=True)
    #     world_size = 1  # 사용 가능한 GPU 수
    #     file_list = self.get_file_list()
    #
    #     for i in range(0, len(file_list), world_size):
    #
    #         current_batch = file_list[i:i + world_size]
    #         processes = []
    #         for j, file_path in enumerate(current_batch):
    #             rank = j % world_size
    #             p = Process(target=self.process_file, args=(rank, file_path))
    #             p.start()
    #             processes.append(p)
    #
    #         for p in processes:
    #             p.join()

    def main(self):
        df = pd.read_csv(os.path.join(self.directory, 'meta_train.csv'), index_col=0)

        X_train, X_val, y_train, y_val = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], shuffle=True, test_size=0.2)
        weight_path = f'Weight/meta_model.pth'

        print('')
        print(f'{weight_path} will be started...')
        time.sleep(2)
        trainer = Run(X_train, y_train, X_val, y_val, weight_path, self.config)
        trainer.load_data()
        trainer.run_model()
        trainer.check_validation()


if __name__ == "__main__":
    E = Execution('Database/', None)
    E.main()
