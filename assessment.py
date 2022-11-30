import os
import glob
import matplotlib.pyplot as plt
from utils import gan_inference_utils

def run_all_epochs(base_name,
                   model_name,
                   model_folder,
                   training_data,
                   validation_data,
                   max_epochs):
    epochs = range(1,max_epochs+1)
    train_result_folders = [None]*len(epochs)
    val_result_folders = [None]*len(epochs)
    for e in epochs:
        train_results = run_gan(base_name+str(e)+'train',
                                training_data,
                                model_name,
                                epoch=epoch)
        validation_results = run_gan(base_name+str(e)+'val',
                                     validation_data,
                                     model_name,
                                     epoch=epoch)
        train_result_folders[e] = train_results
        val_result_folders[e] = val_results
    return train_result_folders, val_result_folders
                          
def assessment_segmentation(base_name,
                            model_name,
                            model_folder,
                            training_data,
                            validation_data,
                            training_results,
                            validation_results,
                            max_epochs):
    training_images = glob.glob(training_data+'\*.jpeg')
    validation_iamges = glob.glob(validation_data+'\*.jpeg')
    validation_scores=np.zeros(max_epochs)
    training_scores = np.zeros(max_epochs)
    epochs = range(1,max_epochs+1)
    for e in epochs:
        k=0
        train_result = training_results[e]
        val_result = validation_results[e]
        dice_scores_training=np.zeros(len(training_images))
        dice_scroes_validation=np.zeros(len(validation_images))
        for i in range(len(training_images)):
            ground_truth = training_images[i]
            name = os.path.basename(ground_truth)
            train_seg = os.path.join(train_result, name)
            
            ground_truth = cv2.imread(ground_truth)
            train_seg = cv2.imread(train_seg)
            train_seg[train_seg>250] = 255
            train_seg[train_seg<255] = 0
            dice_train = np.sum(train_seg[ground_truth==k]==k)*2.0 / (np.sum(train_seg[train_seg==k]==k) + np.sum(ground_truth[ground_truth==k]==k))
            
            dice_scores_training[i] = dice_train
            
        for i in range(len(validation_images)):
            ground_truth = training_images[i]
            name = os.path.basename(ground_truth)
            val_seg = os.path.join(val_result,name)

            ground_truth = cv2.imread(ground_truth)
            val_seg = cv2.imread(val_seg)
            val_seg[val_seg>250] = 255
            val_seg[val_seg<255] = 0
            dice_val = np.sum(val_seg[ground_truth==k]==k)*2.0 / (np.sum(val_seg[val_seg==k]==k) + np.sum(ground_truth[ground_truth==k]==k))
            dice_scores_validation[i] = dice_val
        validation_scores[e] = np.mean(dice_scores_validation)
        training_scores[e] = np.mean(dice_scores_training)


    scores_df = pd.DataFrame({'validation_dice':validation_scores,
                              'training_dice':training_scores,
                              'epoch':epochs}
                             )
    scored_df.to_file(os.path.join(model_folder, 'dice_scores_train_val.csv'), index=False)
    plt.plot(epochs, training_scores, label='Training Data')
    plt.plot(epochs, validation_scores, label='Validation Data')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, 'dice_scores_train_val.png'), dpi=300)
    plt.close()
        


















    
