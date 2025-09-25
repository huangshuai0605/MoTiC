from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from .Network import *

from . import proxy


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        # self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:     #加载checkpoint
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.model.load_state_dict(torch.load(self.args.model_dir)['params'])
        else:                                   #随机初始化参数
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
        
        self.MocoNet = MocoNet(args, self.model.encoder, self.model.classifier)
        self.MocoNet = nn.DataParallel(self.MocoNet, list(range(self.args.num_gpu) ))
        self.MocoNet = self.MocoNet.cuda()
        
                

    def getTrainParam(self):
        params_list = []
        params_list.append( {'params':self.model.encoder.parameters(), 'lr':self.args.lr_base,'lr_min':0.0, 'lr_max':self.args.lr_base,'warmup_epoch':int(self.args.epochs_base*0.05) } )
        params_list.append( {'params':self.model.classifier.parameters(), 'lr':self.args.lr_base,'lr_min':0.0, 'lr_max':self.args.lr_base,'warmup_epoch':int(self.args.epochs_base*0.05) } )
        # params_list.append( {'params':self.model.scale_cos, 'lr':self.args.lr_base,'lr_min':0.0, 'lr_max':self.args.lr_base,'warmup_epoch':int(self.args.epochs_base*0.05) } )
        return params_list
    
    def get_optimizer_base(self):
        if self.args.dataset != 'cub200':
            optimizer = torch.optim.SGD(self.getTrainParam(), lr=self.args.lr_base, weight_decay=self.args.decay, momentum=self.args.momentum, nesterov=True)
        else:   #cub200 微调encoder， 训练分类器
            param_lists = [{'params': self.model.encoder.parameters(), 'lr': self.args.lr_base, 'lr_min':0.0, 'lr_max':self.args.lr_base,'warmup_epoch':int(self.args.epochs_base*0.05) },
                           {'params': self.model.classifier.parameters(), 'lr': 0.1,'lr_min':0.0, 'lr_max':0.1,'warmup_epoch':int(self.args.epochs_base*0.05) }]
            optimizer = torch.optim.SGD(param_lists, weight_decay=self.args.decay, momentum=0.9, nesterov=True)

        return optimizer

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader


    def train(self):
        #torch.set_num_threads(1)
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        # self.transform, self.num_trans = proxy.__dict__["rotation"]()
        self.transform, self.num_trans = proxy.__dict__[args.fantasy]()
        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer = self.get_optimizer_base()
                for epoch in range(args.epochs_base):
                    
                    start_time = time.time()
                    
                    self.MocoNet.module.train_mode()
                    tl, ta = base_train(self.MocoNet.module, trainloader, optimizer, epoch, args, self.transform, self.num_trans)
                    self.MocoNet.module.eval_mode()
                    # tsl, tsa = test(self.model, testloader, epoch, args, session)
                    tsl, tsa = test(self.model,testloader,epoch, self.transform, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        # if epoch>=int(0.5*args.epochs_base):
                        #     self.best_model_dict = deepcopy(self.model.state_dict())

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = optimizer.param_groups[0]['lr']
                    result_list.append(
                        'epoch:%03d,lr:%.4f,ce_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
                
                # self.model.load_state_dict(self.best_model_dict)
                replace_base_fc(train_set, testloader.dataset.transform, self.transform, self.model, args)
                model_dir = os.path.join(args.save_path, 'session' + str(session) + '_acc.pth')
                print('Replace the fc with average embedding, and save it to :%s' % model_dir)
                torch.save(dict(params=self.model.state_dict()), model_dir)

                # self.model.module.mode = 'avg_cos'
                self.model.mode = 'avg_cos'
                tsl, tsa = test(self.model, testloader, 0,self.transform , args, session)
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                if (tsa * 100) >= self.trlog['max_acc'][session]:
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

            else:  # incremental learning sessions
                print("training session: [%d]" % session)
                # import pdb 
                # pdb.set_trace()
                self.model.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                # self.model.update_fc(trainloader, np.unique(train_set.targets), session)    
                self.model.update_fc(trainloader, np.unique(train_set.targets), self.transform, session )
                tsl, tsa, tsa_base, tsa_new = test(self.model, testloader, 0,self.transform ,args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                self.trlog['base_acc'][session] = float('%.3f' % (tsa_base * 100))
                self.trlog['new_acc'][session] = float('%.3f' % (tsa_new * 100))

                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                if session == args.sessions-1:
                    torch.save(dict(params=self.model.state_dict()), save_model_dir)
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        print('-------------Training Results-------------')
        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append('Total Acc.\n{}'.format(self.trlog['max_acc']))
        result_list.append('Base Acc.\n{}'.format(self.trlog['base_acc']))
        result_list.append('New Acc.\n{}'.format(self.trlog['new_acc']))

        print('Total Acc.',self.trlog['max_acc'])
        print('Base Acc.',self.trlog['base_acc'])
        print('New Acc.',self.trlog['new_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        
        #get local time
        timestamp = time.time()
        local_time = time.localtime(timestamp)
        fomatted_time = time.strftime('%Y-%m-%d-%H-%M-%S', local_time)

        self.args.save_path = self.args.save_path + '%s/' % fomatted_time

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None