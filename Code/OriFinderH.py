import torch
import torch.nn as nn
from itertools import product
from Bio import SeqIO
import argparse


BASE_LIST = ['A', 'C', 'G', 'T']


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        Q = self.query(x).unsqueeze(1)
        K = self.key(x).unsqueeze(2)
        V = self.value(x).unsqueeze(2)
        att_weights = torch.matmul(Q, K.transpose(-1, 2))
        att_weights = att_weights / (self.hidden_size ** 0.5)
        att_weights = torch.softmax(att_weights, dim=-1)
        att_output = torch.matmul(att_weights, V).squeeze(2)        
        return att_output


class MainModel(nn.Module):
    def __init__(self, in_channels):
        super(MainModel, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, )),
            nn.ELU(),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, )),
            nn.ELU(),
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, )),
            nn.ELU(),
        )

        self.down_sample_1 = nn.AvgPool1d(kernel_size=(3, ))
        self.down_sample_2 = nn.AvgPool1d(kernel_size=(3, ))
        self.down_sample_3 = nn.AvgPool1d(kernel_size=(3, ))

        self.self_attention_1 = SelfAttention(input_size=12+12+36+48+144, hidden_size=12+12+36+48+144)
        self.self_attention_2 = SelfAttention(input_size=166, hidden_size=166)
        self.self_attention_3 = SelfAttention(input_size=110, hidden_size=110)
        self.self_attention_4 = SelfAttention(input_size=72, hidden_size=72)

        self.output_block = nn.Sequential(
            nn.Linear(in_features=72, out_features=2),
            nn.Softmax(dim=-1)
        )

        self.norm_2 = nn.LayerNorm(166)
        self.norm_3 = nn.LayerNorm(110)
        self.norm_4 = nn.LayerNorm(72)


    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], 1, x.shape[-1]))
        x = self.self_attention_1(x)
        x = self.conv_block_1(x)
        x = self.down_sample_1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)

        x = self.norm_2(x)
        x = self.self_attention_2(x)
        x = self.conv_block_2(x)
        x = self.down_sample_2(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        
        x = self.norm_3(x)
        x = self.self_attention_3(x)
        x = self.conv_block_3(x)
        x = self.down_sample_3(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='linear', align_corners=False)

        x = self.norm_4(x)
        x = self.self_attention_4(x)
        x = self.output_block(x)
        return torch.reshape(x, (x.shape[0], x.shape[-1]))
    

def countBase(sequence_in, target):
    try:
        return sequence_in.count(target) / len(sequence_in)
    except:
        print(sequence_in)


def creatBasePair(pair_number, base_list=None):
    if base_list is None:
        base_list = ['A', 'C', 'G', 'T']
    if pair_number == 1:
        return base_list
    else:
        base_list = [f'{i}{j}' for i, j in product(base_list, BASE_LIST)]
        return creatBasePair(pair_number - 1, base_list)
    

class ZCoding:
    def __init__(self, sequence, z_number) -> None:
        self.base_list = ['A', 'C', 'G', 'T']
        self.z_number = z_number
        self.sequence = sequence
        self.split_sequence = ' '.join([sequence[i:i+z_number] for i in range(len(sequence)-z_number)])

    def coding(self):
        if self.z_number == 1:
            count_list = [countBase(self.split_sequence, base) for base in self.base_list]
            x = (count_list[0] + count_list[2]) - (count_list[1] + count_list[3])
            y = (count_list[0] + count_list[1]) - (count_list[2] + count_list[3])
            z = (count_list[0] + count_list[3]) - (count_list[1] + count_list[2])
            return [x, y, z]
        else:
            result_list = []
            base_list = creatBasePair(self.z_number)
            split_base_list = []
            tmp_base_list = []
            for base in base_list:
                tmp_base_list.append(base)
                if len(tmp_base_list) == 4:
                    split_base_list.append(tmp_base_list)
                    tmp_base_list = []
            for tmp_base_list in split_base_list:
                count_list = [countBase(self.split_sequence, base) for base in tmp_base_list]
                x = (count_list[0] + count_list[2]) - (count_list[1] + count_list[3])
                y = (count_list[0] + count_list[1]) - (count_list[2] + count_list[3])
                z = (count_list[0] + count_list[3]) - (count_list[1] + count_list[2])
                result_list += [x, y, z]
            return result_list
        
    def codingPhase(self):
        result_list = []
        self.phase_sequence_list = []
        for start in range(3):
            target_sequence = self.sequence[start:]
            self.phase_sequence_list.append("".join(target_sequence[i] for i in range(0, len(target_sequence), 3)))
        
        if self.z_number == 1:
            for sequence in self.phase_sequence_list:
                sequence = ' '.join([sequence[i:i+self.z_number] for i in range(len(sequence)-self.z_number)])
                count_list = [countBase(sequence, base) for base in self.base_list]
                x = (count_list[0] + count_list[2]) - (count_list[1] + count_list[3])
                y = (count_list[0] + count_list[1]) - (count_list[2] + count_list[3])
                z = (count_list[0] + count_list[3]) - (count_list[1] + count_list[2])
                result_list += [x, y, z]
        else:
            base_list = creatBasePair(self.z_number)
            split_base_list = []
            tmp_base_list = []
            for base in base_list:
                tmp_base_list.append(base)
                if len(tmp_base_list) == 4:
                    split_base_list.append(tmp_base_list)
                    tmp_base_list = []
            for tmp_base_list in split_base_list:
                for sequence in self.phase_sequence_list:
                    sequence = ' '.join([sequence[i:i+self.z_number] for i in range(len(sequence)-self.z_number)])
                    count_list = [countBase(sequence, base) for base in tmp_base_list]
                    x = (count_list[0] + count_list[2]) - (count_list[1] + count_list[3])
                    y = (count_list[0] + count_list[1]) - (count_list[2] + count_list[3])
                    z = (count_list[0] + count_list[3]) - (count_list[1] + count_list[2])
                    result_list += [x, y, z]
        return result_list
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-fasta", "--fasta_path", help="Sequence file in fasta format", type=str)
    parser.add_argument("-out", "--output_path", help="Result will save in this path",  type=str, default="")
    parser.add_argument("-model_path", "--model_path", help="Please enter model path",  type=str, default="ori_finder_h.pkl")
    args = parser.parse_args()
    sequence_file_path = args.fasta_path
    save_file_path     = args.output_path
    save_file          = open(save_file_path, "w", encoding="UTF-8")
    sequence_list      = list(SeqIO.parse(sequence_file_path, "fasta"))
    ori_finder_h       = torch.load(args.output_path, map_location="cpu")
    for seq_num, seq in enumerate(sequence_list):
        print(f"Test sequence num:[{seq_num+1}/{len(sequence_list)}].")
        coded_list  = []
        coded_list += ZCoding(sequence=str(seq.seq).upper(), z_number=1).coding()
        coded_list += ZCoding(sequence=str(seq.seq).upper(), z_number=1).codingPhase()
        coded_list += ZCoding(sequence=str(seq.seq).upper(), z_number=2).coding()
        coded_list += ZCoding(sequence=str(seq.seq).upper(), z_number=2).codingPhase()
        coded_list += ZCoding(sequence=str(seq.seq).upper(), z_number=3).coding()
        coded_list += ZCoding(sequence=str(seq.seq).upper(), z_number=3).codingPhase()
        coded_list  = torch.Tensor([coded_list])
        test_result = ori_finder_h(coded_list)[0]
        if torch.argmax(test_result) == 1:
            save_file.write(f"{seq.id} is an ORI in human cell. Prop:{test_result[1].item()}\n")
        else:
            save_file.write(f"{seq.id} is not an ORI in human cell. Prop:{test_result[0].item()}\n")
    save_file.close()


if __name__ == '__main__':
    print("Please enter your file like:")
    print("python OriFinderH.py -fasta your_fasta_file_path -out output_path")
    main()