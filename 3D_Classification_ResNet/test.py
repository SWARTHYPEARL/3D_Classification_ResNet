from torch.utils.data import DataLoader
from torchvision import transforms

from ResNet_3D import *
# from datasets import *
from datasets_temp import *

from sklearn import metrics
import matplotlib.pyplot as plt
from scipy import interpolate


def test_model(test_dataset_list: list, model_path_list: list, device: torch.device):
    predict_list = []
    label_list = []
    for idx in range(len(test_dataset_list)):
        test_dataset = test_dataset_list[idx]
        model_path = model_path_list[idx]

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

        model = r3d_18().float()
        # if torch.cuda.device_count() > 1:
        #    print("CUDA Parallel processing is now operating")
        #    model = torch.nn.DataParallel(model)
        # model = torch.nn.DataParallel(model)
        print(device)
        model.to(device)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        activation = nn.Softmax(dim=1).to(device)

        predict_tensorlist = torch.zeros(size=(0, 1), device=torch.device("cpu"))
        label_tensorlist = torch.zeros(size=(0, 1), device=torch.device("cpu"))
        sensitivity_list = []
        specificity_list = []
        sensitivity_easy_list = []
        sensitivity_hard_list = []
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data["target"].to(device), data["class_num"].to(device)
            source = data["source"]
            # print(f"inputs: {inputs}")
            # print(f"labels: {labels}")

            outputs = model(inputs.float())
            outputs_act = activation(outputs)
            # print(outputs_act)

            # _, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.max(outputs_act.data, 1)
            label_truth = torch.tensor([0 if label_list[0] == 1 else 1 for label_list in labels]).to(
                device)  # just for binary class
            # print(outputs_act)
            # print(label_truth)
            # print(outputs_act[range(label_truth.size(0)), label_truth])

            for cmp_idx, cmp_elmt in enumerate(predicted != label_truth):
                if cmp_elmt == 1:
                    print(outputs)
                    print(labels)
                    print(f"Wrong predict: {data['source'][cmp_idx]}\n")
                    if label_truth[cmp_idx] == 1:
                        sensitivity_list.append(0)
                        if "easy" in data["source"][cmp_idx]:
                            sensitivity_easy_list.append(0)
                        elif "hard" in data["source"][cmp_idx]:
                            sensitivity_hard_list.append(0)
                    else:
                        specificity_list.append(0)
                else:
                    if label_truth[cmp_idx] == 1:
                        sensitivity_list.append(1)
                        if "easy" in data["source"][cmp_idx]:
                            sensitivity_easy_list.append(1)
                        elif "hard" in data["source"][cmp_idx]:
                            sensitivity_hard_list.append(1)
                    else:
                        specificity_list.append(1)

            # predict_tensorlist = torch.cat((predict_tensorlist, outputs_act[range(label_truth.size(0)), label_truth].detach().clone().to(device=torch.device("cpu")).view(outputs_act.size(0), 1)))
            predict_tensorlist = torch.cat((predict_tensorlist,
                                            outputs_act[:, 1].detach().clone().to(device=torch.device("cpu")).view(
                                                outputs_act.size(0), 1)))
            label_tensorlist = torch.cat((label_tensorlist,
                                          label_truth.detach().clone().to(device=torch.device("cpu")).view(
                                              label_truth.size(0), 1)))

        # print(predict_tensorlist)
        print(f"Sens-easy[{sum(sensitivity_easy_list)}/{len(sensitivity_easy_list)}]: %0.3f" % (
                    sum(sensitivity_easy_list) / len(sensitivity_easy_list)))
        print(f"Sens-hard[{sum(sensitivity_hard_list)}/{len(sensitivity_hard_list)}]: %0.3f" % (
                    sum(sensitivity_hard_list) / len(sensitivity_hard_list)))
        print(f"Sensitivity[{sum(sensitivity_list)}/{len(sensitivity_list)}]: %0.3f" % (
                    sum(sensitivity_list) / len(sensitivity_list)))
        print(f"Specificity[{sum(specificity_list)}/{len(specificity_list)}]: %0.3f" % (
                    sum(specificity_list) / len(specificity_list)))

        predict_list.append(predict_tensorlist)
        label_list.append(label_tensorlist)

        del model
    # ROC_curve_plot(predict_list, label_list)


def ROC_curve_plot(predict_list: list, label_list: list):
    # sklearn ROC calcalte higher label as positive case

    fig = plt.figure(figsize=(10, 7), dpi=600)

    for idx in range(len(predict_list)):
        predict_tensorlist = predict_list[idx]
        label_tensorlist = label_list[idx]

        # print(metrics.roc_auc_score(label_tensorlist, predict_tensorlist))
        auc_score = metrics.roc_auc_score(label_tensorlist, predict_tensorlist)
        print("AUC: %0.3f" % auc_score)

        fpr, tpr, thresholds = metrics.roc_curve(label_tensorlist, predict_tensorlist)
        # print(fpr)
        # print(tpr)
        # print(thresholds)

        fig = plt.figure(figsize=(10, 7), dpi=600)

        f1 = interpolate.interp1d(fpr, tpr)
        fpr_new = np.linspace(fpr.min(), fpr.max(), num=20, endpoint=True)
        # print(type(fpr_new))
        # print(type(f1(fpr_new)))

        x_final = np.append([0], fpr_new)
        y_final = np.append([0], f1(fpr_new))
        plt.plot(x_final, y_final, color="red", linewidth=2, label="ROC curve (area = %0.3f)" % auc_score)
        plt.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")

        plt.fill_between(x_final, 0, y_final, color="gray", alpha=0.2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

        # plt.show()
        fig.savefig(f"plot_{idx}.png")


if __name__ == "__main__":

    # temp_savepath = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/temp_test"
    # valid_cache_num = 1000
    # isTest = True

    # model_path = "C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/model/model_trained_last.pth"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    test_dataset_list = []
    model_path_list = []
    #for idx in range(1):
    for idx in [0, 5]:
        test_dataset = Tensor3D_Dataset(
            tensor_dir=f"C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/results/training_20210128_2/{idx}/val"
        )
        model_path = f"C:/Users/SNUBH/SP_work/Python_Project/3D_Classification_ResNet/results/training_20210128_2/{idx}/model_trained_last.pth"

        test_dataset_list.append(test_dataset)
        model_path_list.append(model_path)

    test_model(test_dataset_list, model_path_list, device)

    '''
    test_dataset_normal = Dicom3D_CT_Crop_Dataset(
        dicom_dir="D:/billion/training_20210121/Normal_easy",
        label_dir="/gray",
        dicom_HU_level=300,
        dicom_HU_width=2500,
        cache_num=0,
        temp_savepath="D:/billion/training_20210121/temp/Normal_easy",
        isTest=isTest,
        transform=transforms.Compose([
            ToTensor3D(),
            Rescale3D((64, 128, 128))
        ]))

    for model_idx in range(10):
        model_path = f"D:/billion/training_20210121/result/{model_idx}/model_trained_last.pth"

        print(f">>>> Test training model - {model_idx}")

        print("<Normal-easy dataset>")
        test_model(test_dataset_normal, model_path)
    '''