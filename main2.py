import os
import time
import torch
import pandas as pd
from Module.run import Run
from utils.parser import config
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

    def process_file(self, rank, X_train, y_train, X_val, y_val):
        self.setup(rank)

        trainer = Run(X_train, y_train, X_val, y_val, self.config)
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
        STN_list = ['STN001', 'STN002', 'STN003', 'STN004', 'STN005', 'STN006', 'STN007', 'STN008', 'STN009', 'STN010',
                    'STN011', 'STN012', 'STN013', 'STN014', 'STN015', 'STN016', 'STN017', 'STN018', 'STN019', 'STN020']
        directory = 'Database/train/'
        files = sorted(filename for filename in os.listdir(directory) if filename.endswith('.csv'))

        X_val = pd.read_csv('Database/val/X_val_norm.csv', index_col=0)
        y_val = pd.read_csv('Database/val/y_val.csv', index_col=0)

        for i, STN in enumerate(STN_list):
            file_list = [sentence for sentence in files if STN in sentence]
            X_train = pd.read_csv(os.path.join(directory, file_list[0]), index_col=0)
            y_train = pd.read_csv(os.path.join(directory, file_list[1]), index_col=0)
            weight_path = f'Weight/{STN}.pth'

            print('')
            print(f'{STN} will be started...')
            time.sleep(2)
            trainer = Run(X_train, y_train, X_val, y_val, weight_path, self.config)
            trainer.load_data()
            trainer.run_model()
            trainer.check_validation()


if __name__ == "__main__":
    E = Execution('Database/total', None)
    E.main()
