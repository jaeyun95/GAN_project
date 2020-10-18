from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys

import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # optimizer 정의
        optimizer = Adam(0.0002, 0.5)

        # 판별자 정의
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])

        # 생성자 정의
        self.generator = self.build_generator()

        # 생성자의 입력(noise)과 출력(img)
        noise = Input(shape=(self.latent_dim,))
        img = self.generator(noise)

        # 최종 모델(결합된 모델)의 경우 생성자만 학습
        self.discriminator.trainable = False

        # 판별자는 생성된 이미지를 입력으로 받아, 유효성을 검사
        validity = self.discriminator(img)

        # 최종 모델(결합된 모델)은 생성자와 판별자를 쌓아서 만든 모델임(GAN)
        self.model = Model(noise, validity)
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
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

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
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    # 학습 함수
    def train(self, epochs, batch_size=128, sample_interval=50):

        # 데이터셋 로드
        (X_train, _), (_, _) = fashion_mnist.load_data()

        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(1,epochs+1):

            # ---------------------
            #  판별자 학습
            # ---------------------

            # 학습에 사용할 이미지 랜덤으로 선택
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # 노이즈 생성
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 생성자가 이미지 생성
            gen_imgs = self.generator.predict(noise)

            # 판별자 학습
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  생성자 학습
            # ---------------------

            # 노이즈 생성
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 판별자 레이블 샘플을 유효한 것으로 지정
            g_loss = self.model.train_on_batch(noise, valid)

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # 이미지를 저장하는 함수
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
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
    gan = GAN()
    gan.train(epochs=5000, batch_size=32, sample_interval=200)
    gan.save_model()