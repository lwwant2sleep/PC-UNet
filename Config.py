import os
import torch
import time
import ml_collections

save_model = True
tensorboard = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # change this as needed
use_cuda = torch.cuda.is_available()
seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)
cosineLR = True
n_channels = 3
n_labels = 1
epochs = 5000 #1000
img_size = 224
print_frequency = 1
save_frequency = 100
vis_frequency = 100
early_stopping_patience = 100
pretrain = False


#task_name = 'ISIC2018'
#task_name = 'Kvasir-SEG'
task_name = 'Glas'
#task_name = 'BUSI'


n_filts = 32
learning_rate = 1e-3
batch_size = 1


model_name = 'ACC_UNet'
#model_name = 'UNetPP'

#model_name = 'SwinUnet'
#model_name = 'SMESwinUnet'
#model_name = 'UCTransNet'
#model_name = 'UNet_base'
#model_name = 'MultiResUnet1_32_1.67'
#model_name = 'efficientunet'


test_session = "session"
session_name = 'session'


train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
if task_name == 'GlaS':
    val_dataset = './datasets/'+ task_name+ '/Test_Folder/'
else :
    val_dataset = './datasets/'+ task_name+ '/Val_Folder/'

save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'




##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    config.patch_sizes = [16,8,4,2]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config





