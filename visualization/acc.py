import matplotlib.pyplot as plt
import argparse
import os
import datetime

def test_anly(axs, log_path, path, draw_length, x):
    test_log_path = path+"/"+log_path+"/test.log" 
    server_log_path = path+"/"+log_path+"/server.log" 

    if os.path.exists(test_log_path) == False:
        return None, None, None, None

    test_log = open(test_log_path)
    server_log = open(server_log_path)
    start_time = None
    test_time_list = []
    test_accuracy_list = []
    test_loss_list = []
    test_gap_list = []
    start_line = None

    for line in server_log:
        if "client_number" in line:
            client_num = line.split(",")
            for item in client_num:
                if "client_number" in item:
                    client_num = int(item.split(" client_number=")[1])
            start_line = "send global model to rank " + str(client_num-1) +"."
        if start_line == None:
            continue
        if start_line in line:
            start_time = line.split("] [")[0].strip("[]")
            start_time = datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
            break            
    for line in server_log:
        if "test_gap" in line:
            test_gap = float(line.split("test_gap: ")[-1])
            test_gap_list.append(test_gap)   
    for line in test_log:
        if "accuracy" in line:
            time = line.split("] [")[0].strip("[]")
            if "." in time:
                time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
            else:
                time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
            time = time-start_time
            accuracy = float(line.split(";")[1].split(" ")[-1])
            loss = float(line.split(";")[2].split(" ")[-1])
            
            test_time_list.append(time.total_seconds())
            test_accuracy_list.append(accuracy)
            test_loss_list.append(loss)

    if x == 0:
        if len(test_accuracy_list) > draw_length:
            test_accuracy_list = test_accuracy_list[:draw_length]
            test_loss_list = test_loss_list[:draw_length]
        if len(test_time_list) > draw_length:
            test_time_list = test_time_list[:draw_length]
    else:
        idx = len(test_time_list)
        for i in range(len(test_time_list)):
            if test_time_list[i] > draw_length:
                idx = i
                break
        if len(test_accuracy_list) > idx:
            test_accuracy_list = test_accuracy_list[:idx]
            test_loss_list = test_loss_list[:idx]
        if len(test_time_list) > idx:
            test_time_list = test_time_list[:idx]
    
    if x == 0:
        if len(test_gap_list) > draw_length:
            test_gap_list = test_gap_list[:draw_length]

    return test_accuracy_list, test_time_list, test_gap_list, test_loss_list


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--length', type=int, default=100)
    parser.add_argument('--x', type=int, default=0)
    parser.add_argument('--avg_step', type=int, default=10)
    parser.add_argument('--top1', type=int, default=0)
    parser.add_argument('--mod', type=int, default=0)
    args = parser.parse_args()
    
    path = args.path
    draw_length = args.length
    x = args.x
    avg_step = args.avg_step
    top1 = args.top1
    mod = args.mod

    if mod != 0:
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,12))
    else:
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
        axs = [axs]

    for log_path in os.listdir(path):  
        test_accuracy_list, test_time_list, test_gap_list, test_loss_list = test_anly(axs, log_path, path, draw_length, x)  

        if test_accuracy_list != None and test_time_list != None and test_gap_list != None:
            max_accuracy = max(test_accuracy_list)
            max_accuracy_idx = test_accuracy_list.index(max_accuracy)
            current_test_gap = test_gap_list[-1]
            avg_test_gap = sum(test_gap_list)/len(test_gap_list)
            print("\nexperiment: %s\nmax_acc: %.4f @ max_acc_idx: %d/%d\ncurrent_acc: %.4f current_test_gap: %.4f\navg_test_gap: %.4f\ntotal_time_cost: %.4f" % (log_path, max_accuracy, max_accuracy_idx, len(test_accuracy_list), test_accuracy_list[-1], current_test_gap, avg_test_gap, test_time_list[-1]))

            top1_list = []
            top1_idx = []
            if top1 == 1:
                top1acc = test_accuracy_list[0]
                for i in range(len(test_accuracy_list)):
                    if test_accuracy_list[i] > top1acc:
                        top1acc = test_accuracy_list[i]
                        top1_list.append(top1acc)
                        top1_idx.append(i)
                    else:
                        test_accuracy_list[i] = top1acc
            if x == 0:
                axs[0].plot(range(len(test_accuracy_list)), test_accuracy_list, label=log_path)
            else:
                axs[0].plot(test_time_list, test_accuracy_list, label=log_path)

            if mod != 0:
                axs[1].plot(range(len(test_gap_list)), test_gap_list, label=log_path)
                axs[2].plot(range(len(test_loss_list)), test_loss_list, label=log_path)

        
            top1_list.insert(0, 0)
            top1_idx.insert(0, 0)
            if len(top1_list) < 2:
                continue
            top1_k = []
            for i in range(1, len(top1_idx)):
                acc_inc = top1_list[i] - top1_list[i-1]
                idx = top1_idx[i] - top1_idx[i-1]
                k = acc_inc/idx
                top1_k.append(k)

            converge = False
            idx = test_accuracy_list.index(top1_list[-1])
            for i in range(1, avg_step):
                if idx + avg_step + 1 > len(test_accuracy_list):
                    converge = False
                    break
                if test_accuracy_list[idx+i] <= test_accuracy_list[idx]:
                    converge = True
                else:
                    converge = False
                    break
            if converge:
                print("converge at: %d, accuracy: %.4f, time: %.4f" % (idx, test_accuracy_list[idx], test_time_list[idx]))
            print("max loss: %.4f, min loss: %.4f" % (max(test_loss_list), min(test_loss_list)))

    if x == 0:
        axs[0].set_xlabel("round")
    else:
        axs[0].set_xlabel("time(s)")
    axs[0].set_ylabel("accuracy")
    #axs[0].set_title(model)
    axs[0].grid(True)
    axs[0].legend()
    if mod != 0:
        axs[1].grid(True)
        axs[1].legend()
        axs[1].set_xlabel("round")
        axs[1].set_ylabel("round time(s)")
        axs[2].legend()
        axs[2].grid(True)

    plt.tight_layout()
    if x == 0:
        plt.savefig("accuracy_byRound.png")
    else:
        plt.savefig("accuracy_byTime.png")