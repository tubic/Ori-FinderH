from itertools import product


BASE_LIST = ['A', 'C', 'G', 'T']


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