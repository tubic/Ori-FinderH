import torch 
from torch import nn
from torch import optim
import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, roc_curve
from matplotlib import pyplot as plt


class TestUnit:
    def __init__(self, model, var_set, auc_path) -> None:
        self.model = model
        self.var_set = var_set
        self.auc_path = auc_path
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            label_from_model = []
            prop_from_model = []
            real_label = []
            for (inputs, labels) in self.var_set:
                inputs = inputs
                labels = labels
                outputs = self.model(inputs)
                outputs = outputs.cpu().tolist()
                label_from_model += [np.argmax(i) for i in outputs]
                prop_from_model += [i[1] for i in outputs]
                real_label += [np.argmax(i) for i in labels.cpu().tolist()]
          
            test_acc = sum([1 for i, j in zip(label_from_model, real_label) if i == j]) / len(real_label)
            test_auc = roc_auc_score(real_label, prop_from_model, multi_class='ovr')
            test_mcc = matthews_corrcoef(real_label, label_from_model)
            test_f1 = f1_score(real_label, label_from_model)
            fpr, tpr, _ = roc_curve(real_label, label_from_model)
            
            plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % test_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.savefig(self.auc_path)
            plt.close()
            return test_acc, test_mcc, test_f1, test_auc



class TestUnitS:
    def __init__(self, model, var_set) -> None:
        self.model = model
        self.var_set = var_set
    
    def test(self):
        self.model.eval()
        with torch.no_grad():
            label_from_model = []
            prop_from_model = []
            real_label = []
            for (inputs, labels) in self.var_set:
                inputs = inputs
                labels = labels
                outputs = self.model(inputs)
                outputs = outputs.cpu().tolist()
                label_from_model += [np.argmax(i) for i in outputs]
                prop_from_model += [i[1] for i in outputs]
                real_label += [np.argmax(i) for i in labels.cpu().tolist()]
          
            test_acc = sum([1 for i, j in zip(label_from_model, real_label) if i == j]) / len(real_label)
            test_mcc = matthews_corrcoef(real_label, label_from_model)
            return test_acc, test_mcc


class TrainUnit:
    def __init__(self, model, loss_function, learning_rate, train_set, var_set, train_step, train_log, auc_path) -> None:
        self.model = model
        self.loss_function = loss_function
        self.model_opt = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.train_set = train_set
        self.var_set = var_set
        self.train_step = train_step
        self.train_log = train_log
        self.auc_path = auc_path

    def train(self):
        for train_step in range(self.train_step):
            mean_loss = 0.
            count_set = 0
            for (train_data, train_label) in self.train_set:
                count_set += 1
                model_result = self.model(train_data) 
                model_loss = self.loss_function(model_result, train_label)
                self.model_opt.zero_grad()
                model_loss.backward()
                self.model_opt.step()
                mean_loss += model_loss.item()
            mean_loss /= count_set
            test_acc, test_mcc, test_f1, test_auc = TestUnit(self.model, self.var_set, self.auc_path).test()
            print(f"Train step:[{train_step+1}/{self.train_step}], Train loss:{mean_loss:.4f}, ACC:{test_acc:.4f}, MCC:{test_mcc:.4f}, F1:{test_f1:.4f}, AUC:{test_auc:.4f}")
            torch.save(self.model, f"{train_step}.pkl")
            self.train_log.write(f"Train step:[{train_step+1}/{self.train_step}], Train loss:{mean_loss:.4f}, ACC:{test_acc:.4f}, MCC:{test_mcc:.4f}, F1:{test_f1:.4f}, AUC:{test_auc:.4f}\n")