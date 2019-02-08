#!/usr/bin/python

import argparse
import os
import pickle

import sys
import base64

import tensorflow as tf

word2vec = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'word2vec_ops.so'))

embeddings = {}


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--embed_pickle_path', dest='embed_pickle_path', help='The file saving all training parameters', type=str, required=True)
    parser.add_argument('-i', '--int2insn_map_path', dest='int2insn_path', help='The pickle file saving int -> instruction mapping.', type=str, required=False, default='int2insn.map')
    parser.add_argument('-m', '--model_path', dest='model_path', help='The file saving the trained embedding model', type=str, required=True)
    parser.add_argument('-o', '--output_file', dest='output_file', help='The file saving the embedding vector for each instruction', type=str, required=False, default='embed.pkl')

    args = parser.parse_args()

    config_info = {
        'embed_pickle_path': args.embed_pickle_path,
        'model_path': args.model_path,
        'output_file': args.output_file,
        'int2insn_path': args.int2insn_path
    }

    return config_info


def main():

    config_info = get_config()

    embed_pickle_path = config_info['embed_pickle_path']
    model_path = config_info['model_path']
    output_file = config_info['output_file']
    int2insn_path = config_info['int2insn_path']

    with tf.Graph().as_default(), tf.Session() as sess:
        print("Loading model...")
        saver = tf.train.import_meta_graph(model_path + ".meta")
        a = saver.restore(sess, model_path)
        print("Model loaded")

        print("Loading embed input data...")
        input_data = pickle.load(open(embed_pickle_path, 'rb'), encoding='latin1')
        print("Embed input data loaded")

        print("Loading int to instruction map...")
        int2insn_map = pickle.load(open(int2insn_path, 'rb'), encoding='latin1')
        int2insn_map['UNK'] = 'UNK'
        print("Int to instruction map loaded")

        w_out = [v for v in tf.global_variables() if v.name == "w_out:0"][0]

        num = 0
        total_num = len(input_data['word2id'])

        ids = []
        vectors = {}

        error_num = 0
        for word in input_data['word2id']:
            word_id = input_data['word2id'][word]
            ids.append(word_id)

            if len(ids) == 1000:
                part_vector = tf.nn.embedding_lookup(w_out, ids).eval()

                for i in range(len(ids)):
                    word_id = ids[i]
                    word = input_data['id2word'][word_id]
                    if word != 'UNK':
                        word = int(word)

                    vector = part_vector[i]
                    insn = int2insn_map[word]
                    embeddings[str(insn)] = {'vector': vector}

                ids = []

                num += 1000
                if num % 1000 == 0:
                    print("{} computed ({}%)".format(num, 100.0 * num / total_num))


        if len(ids) > 0:
            part_vector = tf.nn.embedding_lookup(w_out, ids).eval()
            for i in range(len(ids)):
                word_id = ids[i]
                word = input_data['id2word'][word_id].decode('utf-8')
                if word != 'UNK':
                    word = int(word)

                vector = part_vector[i]
                insn = int2insn_map[word]
                embeddings[str(insn)] = {'vector': vector}


        print("{} Errors".format(error_num))
        pickle.dump(embeddings, open(output_file, "wb"))
        print("Done")

if __name__ == '__main__':
    main()
