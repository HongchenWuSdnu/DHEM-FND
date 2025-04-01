import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import logging
import json
from utils.dataloader import BertDataLoader
from models.dhem_fnd import Trainer as DHEMFNDTrainer

def frange(x, y, jump):
  while x < y:
      x = round(x, 8)
      yield x
      x += jump

class Run():
    def __init__(self,
                 config
                 ):
        self.configinfo = config

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.batchsize = config['batchsize']
        self.emb_dim = config['emb_dim']
        self.weight_decay = config['weight_decay']
        self.lr = config['lr']
        self.epoch = config['epoch']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.early_stop = config['early_stop']
        self.root_path = config['root_path']
        self.mlp_dims = config['model']['mlp']['dims']
        self.dropout = config['model']['mlp']['dropout']
        self.seed = config['seed']
        self.save_log_dir = config['save_log_dir']
        self.save_param_dir = config['save_param_dir']
        self.param_log_dir = config['param_log_dir']

        self.semantic_num = config['semantic_num']
        self.emotion_num = config['emotion_num']
        self.style_num = config['style_num']
        self.lnn_dim = config['lnn_dim']
        self.domain_num = config['domain_num']
        self.category_dict = config['category_dict']
        self.dataset = config['dataset']


        self.train_path = self.root_path + 'train.pkl'
        self.val_path = self.root_path + 'val.pkl'
        self.test_path = self.root_path + 'test.pkl'
        
    
    def get_dataloader(self):
        loader = BertDataLoader(max_len = self.max_len, batch_size = self.batchsize,
                        category_dict = self.category_dict, num_workers=self.num_workers, dataset_type = self.dataset)
        train_loader = loader.load_data(self.train_path, True)
        val_loader = loader.load_data(self.val_path, False)
        test_loader = loader.load_data(self.test_path, False)
        return train_loader, val_loader, test_loader

    def getFileLogger(self, log_file):
        logger = logging.getLogger()
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    def config2dict(self):
        config_dict = {}
        for k, v in self.configinfo.items():
            config_dict[k] = v
        return config_dict

    def main(self):
        param_log_dir = self.param_log_dir
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir)
        param_log_file = os.path.join(param_log_dir, self.model_name +'_'+ 'oneloss_param.txt')
        logger = self.getFileLogger(param_log_file)
        self.model_name == 'DHEM-FND'
        train_loader, val_loader, test_loader = self.get_dataloader()

        if self.model_name == 'DHEM-FND':
            trainer = DHEMFNDTrainer(
                emb_dim=self.emb_dim,
                mlp_dims=self.mlp_dims,
                use_cuda=self.use_cuda,
                lr=self.lr,
                dropout=self.dropout,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                category_dict=self.category_dict,
                weight_decay=self.weight_decay,
                save_param_dir=os.path.join(self.save_param_dir, self.model_name),
                semantic_num=self.semantic_num,
                emotion_num=self.emotion_num,
                style_num=self.style_num,
                lnn_dim=self.lnn_dim,
                dataset=self.dataset,
                early_stop=self.early_stop,
                epoches=self.epoch
            )
            train_param = {
            'lr': [self.lr] * 10
        }
        print(train_param)
        param = train_param
        best_param = []

        json_path = './logs/json/' + self.model_name +'.json'
        json_result = []
        for p, vs in param.items():
            best_metric = {}
            best_metric['metric'] = 0
            best_v = vs[0]
            best_model_path = None
            for i, v in enumerate(vs):
                setattr(trainer, p, v)
                # 调用绘图方法
                #trainer.visualize_features()
                metrics, model_path = trainer.train(logger)
                json_result.append(metrics)
                if(metrics['metric'] > best_metric['metric']):
                    best_metric = metrics
                    best_v = v
                    best_model_path = model_path
            best_param.append({p: best_v})
            print("best model path:", best_model_path)
            print("best metric:", best_metric)
            logger.info("best model path:" + best_model_path)
            logger.info("best param " + p + ": " + str(best_v))
            logger.info("best metric:" + str(best_metric))
            logger.info('--------------------------------------\n')
        with open(json_path, 'w') as file:
            json.dump(json_result, file, indent=4, ensure_ascii=False)


