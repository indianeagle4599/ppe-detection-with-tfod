import os
from object_detection.utils import dataset_util, label_map_util
import tensorflow.compat.v1 as tf
from PIL import Image
import io
import pandas as pd
from collections import namedtuple

label_map_dict = {1:'Helmet', 2:'Vest', 3:'Worker'}

def txt_to_csv(data):
    path = f'worskpace/training_demo/pictor/Labels/pictor_ppe_crowdsourced_approach-01_{data}.txt'

    f = open(path, 'r')
    text = f.read().split('\n')
    
    data_frame = []
    for data in text:
        datapoint = data.split('\t')
        
        filename = datapoint[0]
        for i in datapoint[1:]:
            xmin, ymin, xmax, ymax, label = i.split(',')
    
            value = (filename,
                    label_map_dict[int(label)+1],
                    int(xmin),
                    int(ymin),
                    int(xmax),
                    int(ymax)
                    )
            data_frame.append(value)
    return pd.DataFrame(data_frame, columns=['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])


def class_text_to_int(row_label):
    return list(label_map_dict.values()).index(row_label)


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format('Images/'+group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def main():
    data = 'test'
    data = 'train'
    output_path = f'worskpace/training_demo/pictor/{data}.record'
    path = 'worskpace/training_demo/pictor/'
    writer = tf.python_io.TFRecordWriter(output_path)
    
    examples = txt_to_csv(data)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(output_path))


if __name__ == "__main__":
    main()