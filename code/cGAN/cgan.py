from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import sys

import numpy as np

class CGAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.num_classes = 10

        # optimizer 정의
        optimizer = Adam(0.0002, 0.5)

        # 판별자 정의
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

        # 생성자 정의
        self.generator = self.build_generator()

        # 생성자의 입력(noise)과 출력(img)
        noise = Input(shape=(self.latent_dim,))
        #### (1) 라벨 선언
        #### hint: 선언 방법은 noise와 동일하며, 크기는 1입니다.
        label = ????

        #### (2) 생성자의 입력은 noise 뿐만 아니라 label도 함께 입력
        #### hint: noise와 label을 하나의 리스트로 모두 입력받습니다.
        #### 입력으로 ex1, ex2모두 받음 --> [ex1, ex2]
        img = self.generator(????)

        # 최종 모델(결합된 모델)의 경우 생성자만 학습
        self.discriminator.trainable = False

        # 판별자는 생성된 이미지를 입력으로 받아, 유효성을 검사
        #### (3) 판별자의 입력 또한 img, label을 합쳐서 사용
        #### hint: img와 label을 하나의 리스트로 모두 입력받습니다.
        validity = self.discriminator(????)

        # 최종 모델(결합된 모델)은 생성자와 판별자를 쌓아서 만든 모델임(GAN)
        #### (4) 최종 모델(GAN)의 입력 또한 noise와 label을 함께 사용
        #### hint: noise와 label을 하나의 리스트로 모두 입력받습니다.
        self.model = Model(????, validity)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # 생성자 정의 함수
    def build_generator(self):

        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()

        #### (5) 이부분도 위와 동일하게 label선언
        #### dtype은 'int32'로 설정합니다.
        noise = Input(shape=(self.latent_dim,))
        label = ????

        # 라벨을 임베딩함
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        #### (6) 임베딩한 라벨을 noise와 요소곱을 통해 model_input을 생성
        #### hint: multiply를 사용하여 noise, label_embedding의 요소곱을 수행합니다.
        model_input = ????

        img = model(model_input)

        return Model([noise,label], img)

    # 판별자 정의 함수
    def build_discriminator(self):

        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        #### (7) 이부분도 위와 동일하게 label선언
        #### dtype은 'int32'로 설정합니다.
        img = Input(shape=self.img_shape)
        label = ????

        # 라벨을 임베딩함
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))

        # 이미지 크기 변경 (28, 28, 1) --> (784)
        flat_img = Flatten()(img)

        #### (8) 임베딩한 라벨을 flat_img 요소곱을 통해 model_input을 생성
        #### hint: multiply를 사용하여 flat_img, label_embedding의 요소곱을 수행합니다.
        model_input = ????
        validity = model(model_input)

        return Model([img, label], validity)

    # 학습 함수
    def train(self, epochs, batch_size=128, sample_interval=50):

        # 데이터셋 로드
        (X_train, y_train), (_, _) = fashion_mnist.load_data()

        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        D_loss_list = []
        G_loss_list = []
        for epoch in range(1,epochs+1):

            # ---------------------
            #  판별자 학습
            # ---------------------

            # 학습에 사용할 이미지 랜덤으로 선택
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # 노이즈 생성
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 생성자가 이미지 생성
            #### (9) 입력으로 noise와 labels을 함께 사용
            #### hint: noise와 labels을 하나의 리스트로 모두 입력받습니다.
            gen_imgs = self.generator.predict(????)

            # 판별자 학습
            #### (10) 입력으로 imgs와 labels을 함께 사용
            #### hint: imgs와 labels을 하나의 리스트로 모두 입력받습니다.
            d_loss_real = self.discriminator.train_on_batch(????, valid)

            #### (11) 입력으로 gen_imgs와 labels을 함께 사용
            #### hint: gen_imgs와 labels을 하나의 리스트로 모두 입력받습니다.
            d_loss_fake = self.discriminator.train_on_batch(????, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  생성자 학습
            # ---------------------

            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 판별자 레이블 샘플을 유효한 것으로 지정
            #### (12) 입력으로 noise와 sampled_labels을 함께 사용
            #### hint: noise와 sampled_labels을 하나의 리스트로 모두 입력받습니다.
            g_loss = self.model.train_on_batch(????, valid)
            G_loss_list.append(g_loss)
            D_loss_list.append(d_loss[0])
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        self.plotLoss(G_loss_list, D_loss_list, epoch)

    # 그래프를 생성하는 함수
    def plotLoss(self, G_loss, D_loss, epoch):
        plt.figure(figsize=(10, 8))
        plt.plot(D_loss, label='Discriminitive loss')
        plt.plot(G_loss, label='Generative loss')
        plt.xlabel('BatchCount')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_graph/gan_loss_epoch_%d.png' % epoch)

    # 이미지를 저장하는 함수
    def sample_images(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
                axs[i,j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    # 모델을 로드하는 함수
    def load_model(self, model_path='saved_model/model.h5'):
        print('\nload model : \"{}\"'.format(model_path))
        self.model = tf.keras.models.load_model(model_path)

    # 모델을 저장하는 함수
    def save_model(self, model_path='saved_model/model.h5'):
        print('\nsave model : \"{}\"'.format(model_path))
        self.model.save(model_path)

if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=5000, batch_size=32, sample_interval=200)
    cgan.save_model()