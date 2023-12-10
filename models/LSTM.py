import torch.nn as nn
import torch.nn.functional as F

class LSTM_KWS(nn.Module):

    def __init__(self):
        super(LSTM_KWS, self).__init__()
        self.rnn = nn.LSTM(
            # input_size=28,
            input_size=10,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(64,10)

    def forward(self,x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:,-1,:])
        return out
    
if __name__=="__main__":
    model = LSTM_KWS()
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size+buffer_size)/1024/1024
    print(all_size, "MB")
    print(param_size, param_sum, buffer_size, buffer_sum, all_size)
    
