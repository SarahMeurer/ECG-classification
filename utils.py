# Python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, Dense, ReLU, Add
from keras.models import Model

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

import seaborn as sns
import pathlib

import plot_utils as putils

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# Rajpurkar model functions
# Create residual blocks
def residual_blocks_rajpurkar(x,i=0, stride=1, num_filter=64, rate_drop=0.5, initializer='none'):

    bn_1 = BatchNormalization()(x)
    relu_1 = ReLU()(bn_1)
    drop_1 = Dropout(rate_drop)(relu_1)
    conv_1 = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer)(drop_1)
    bn_2 = BatchNormalization()(conv_1)
    relu_2 = ReLU()(bn_2)
    drop_2 = Dropout(rate_drop)(relu_2)
    conv_2 = Conv1D(kernel_size=16, filters=num_filter, strides=stride, padding="same", kernel_initializer=initializer)(drop_2)

    if i == 3 or i == 7 or i == 11:  #Verifica se houve mudança na quantidade de número de filtros
        #Short connection
        conv_aj = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same")(x) #Ajustar o número de filtros
        short = MaxPooling1D(pool_size = 1, strides=2)(conv_aj)
    else:
        #Short connection
        short = MaxPooling1D(pool_size = 1, strides=stride)(x)

    # Adding layers
    return Add()([conv_2, short])


# Ribeiro's model functions
# Create the skip connection A with MaxPooling and Conv layers
def skip_connection(skip, num_filter=128, rate_drop=0, initializer='none', downsample=1):
    skip = MaxPooling1D(pool_size=downsample,strides=downsample,padding='same')(skip)
    skip = Conv1D(filters=num_filter,kernel_size=1,strides=1,padding='same')(skip)
    return skip

# Create the residual blocks
def residual_blocks_ribeiro(input, num_filter=128, rate_drop=0, initializer='none', downsample=1):

    layer, skip = input

    skip = skip_connection(skip, num_filter=num_filter, rate_drop=rate_drop, initializer=initializer, downsample=downsample)

    layer = Conv1D(kernel_size=16, filters=num_filter, strides=1, padding="same", kernel_initializer=initializer)(layer) 
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Dropout(rate_drop)(layer)
    layer = Conv1D(kernel_size=16, filters=num_filter, strides=downsample, padding="same", kernel_initializer=initializer)(layer) 

    layer = Add()([layer, skip])
    skip = layer

    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Dropout(rate_drop)(layer)

    return layer, skip

# Get the models os the network
def get_model(input_layer, model_name):

    if 'rajpurkar' in model_name:
        rate_drop = 1 - 0.8
        initializer = 'he_normal'

        conv_config = dict(
            kernel_size=16, filters=64,
            padding="same", kernel_initializer=initializer
        )

        # First layer
        conv_1 = Conv1D(strides=1, **conv_config)(input_layer)
        bn_1 = BatchNormalization()(conv_1)
        relu_1 = ReLU()(bn_1)

        # Second layer
        conv_2 = Conv1D(strides=1, **conv_config)(relu_1)
        bn_2 = BatchNormalization()(conv_2)
        relu_2 = ReLU()(bn_2)
        drop_1 = Dropout(rate_drop)(relu_2)
        conv_3 = Conv1D(strides=2, **conv_config)(drop_1)

        # Short connection
        short_1 = MaxPooling1D(pool_size=1, strides=2)(relu_1)

        # Adding layers
        layers = Add()([conv_3, short_1])

        num_filter = [
            64, 64, 64,
            128, 128, 128, 128,
            192, 192, 192, 192,
            256, 256, 256, 256
        ]
        for i in range(15):
            #print(f"i = {i} STRIDE = {(i % 2)+1}, FILTER LENGHT = {num_filter[i]}")
            layers = residual_blocks_rajpurkar(
                layers, i=i, stride=(i % 2)+1, num_filter=num_filter[i],
                rate_drop=rate_drop, initializer=initializer
            )

        # Last layers
        # The ﬁnal fully connected layer and sigmoid activation produce a distribution 
        # over the 5 output superclasses for each time-step.
        bn_x = BatchNormalization()(layers)
        relu_x = ReLU()(bn_x)
        flat_x = Flatten()(relu_x)
        dense_x = Dense(32)(flat_x)
        classification = Dense(5, activation='sigmoid')(dense_x)
    
    elif 'ribeiro' in model_name:
        initializer = 'he_normal'
        rate_drop = 1 - 0.8
        downsample = 4

        # Input block
        layers = Conv1D(
            kernel_size=16, filters=64, strides=1, padding="same", kernel_initializer=initializer
        )(input_layer)  # Output_size = (1000, 64)
        layers = BatchNormalization()(layers)
        layers = ReLU()(layers)

        num_filter = np.array([128, 192, 256, 320])

        layer = layers
        skip = layers

        # Residual Blocks
        for i in range(4):
            layer, skip = residual_blocks_ribeiro(
                [layer, skip], num_filter=num_filter[i], rate_drop=rate_drop, initializer=initializer, downsample=downsample
            )

        # Output block
        layer = Flatten()(layer)
        classification = Dense(5, activation='sigmoid',
                               kernel_initializer=initializer)(layer)
    else:
        print('Wrong name')
        return

    # Constructing the model
    model = Model(inputs=input_layer, outputs=classification)

    return model

# Get the metrics
def get_metrics(y_test, prediction, prediction_bin, target_names):

    print("\nReports from the classification:")

    report = classification_report(y_test,prediction_bin,output_dict=True,target_names=target_names)
    report_df = pd.DataFrame.from_dict(report, orient='index')
    print(report_df)

    roc_auc_macro = roc_auc_score(y_test, prediction, average='macro')
    roc_auc_micro = roc_auc_score(y_test, prediction, average='micro')
    roc_auc_weighted = roc_auc_score(y_test, prediction, average='weighted')
    roc_auc_none = roc_auc_score(y_test, prediction, average=None)

    print(f'ROC AUC macro = {roc_auc_macro}')
    print(f'ROC AUC micro = {roc_auc_micro}')
    print(f'ROC AUC weighted = {roc_auc_weighted}')
    print(f'ROC AUC None = {roc_auc_none}')

    return

# Plot results
def plot_results(history, name, metric, plot_path='plots'):

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(history.epoch, history.history[metric], '-o')
    ax.plot(history.epoch, history.history[f'val_{metric}'], '-*')

    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{metric}'.capitalize())
    ax.legend(['Training set', 'Validation set'])

    # Save figure in multiple formats
    filename = plot_path / f'[{name}][{metric}]'
    fig.savefig(f'{filename}.png', format='png', dpi=600)
    fig.savefig(f'{filename}.pdf', format='pdf')

    # # Plot results
    # fig, ax = plt.subplots()
    # ax.plot(history.epoch, history.history['accuracy'], '-o')
    # ax.plot(history.epoch, history.history['val_accuracy'], '-*')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Accuracy')
    # ax.legend(['Training set', 'Validation set'])

    # # Save figure in multiple formats
    # filename = f'{plot_path}/[{model_name}][Accuracy][100]'
    # fig.savefig(f'{filename}.png', format='png', dpi=600)
    # fig.savefig(f'{filename}.pdf', format='pdf')

    return

# Transform the values in the confusion matrix in percentage
def get_cm_percent(cm, total):
    cm_perc = np.zeros_like(cm, dtype='float')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            for k in range(cm.shape[2]):
                cm_perc[i][j][k] = round((cm[i][j][k] / total) * 100, 2)
    return cm_perc

# Plot confusion matrix


def plot_confusion_matrix(y_test, y_pred, model_name, target_names, plot_path='plots'):

    # Make sure the plot folder exists
    plot_path = pathlib.Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    sample_weight = compute_sample_weight(
        class_weight='balanced', y=y_test)

    # Confusion matrix
    cm = multilabel_confusion_matrix(y_test, y_pred)
    cm_perc = get_cm_percent(cm=cm, total=len(y_pred))

    print(f'\n{cm}')
    # print(f'\n{cm_perc}')

    # Plot confusion matrix
    fig_list = []
    for i, (label, matrix) in enumerate(zip(target_names, cm_perc)):
        fig, ax = plt.subplots()

        labels = [f'Não é {label}', label]
        sns.heatmap(matrix, annot=True, square=True, fmt='.2f', cbar=False, cmap='Blues',
                         xticklabels=labels, yticklabels=labels, linecolor='black', linewidth=1, ax=ax)
        for t in ax.texts:
            t.set_text(t.get_text() + '%')
        ax.set_xlabel('Rótulo predito')
        ax.set_ylabel('Rótulo verdadeiro')
        fig.tight_layout()

        fig_folder = plot_path / f'figures-{model_name}'
        name = f'confusion_matrix-{label}'
        putils.save_fig(fig, name, path=fig_folder, figsize='square', usetex=False)

        # fig.savefig(f'{filename}.png', format='png', dpi=600)
        # fig.savefig(f'{filename}.pdf', format='pdf')
        fig_list.append(fig)

    return fig_list


