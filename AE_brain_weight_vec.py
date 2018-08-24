"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
matplotlib
numpy
"""
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.utils.data as Data
import numpy as np
import copy
import cPickle as pickle
import argparse
from ranking import *

torch.manual_seed(1)    # reproducible

def evaluation(ep, text, image, sound, autoencoder, vocab, args):
    testfile = ['men-3k.txt', 'simlex-999.txt', 'semsim.txt', 'vissim.txt', 'simverb-3500.txt',
                'wordsim353.txt', 'wordrel353.txt', 'association.dev.txt', 'association.dev.b.txt']
    _, _,_,_, multi_rep = autoencoder(text, image, sound)
    word_vecs = multi_rep.data.cpu().numpy()
    #torch.save(autoencoder.state_dict(), open(args.outmodel + '.parameters-' + str(ep), 'wb'))
    #outfile = open(args.outmodel + '-' + str(ep)+ '.rep.txt', 'w')
    # outfile = open(args.outmodel+'.rep.txt', 'w')
    # #pickle.dump(word_vecs, outfile, protocol=2)
    # for ind, w in enumerate(word_vecs):
    #     outfile.write(vocab[ind] + ' ' + ' '.join([str(i) for i in w]) + '\n')

    for file in testfile:
        manual_dict, auto_dict = ({}, {})
        not_found, total_size = (0, 0)
        for line in open('evaluation/' + file, 'r'):
            line = line.strip().lower()
            word1, word2, val = line.split()
            if word1 in vocab and word2 in vocab:
                manual_dict[(word1, word2)] = float(val)
                auto_dict[(word1, word2)] = cosine_sim(word_vecs[vocab.index(word1)],
                                                       word_vecs[vocab.index(word2)])
            else:
                not_found += 1
            total_size += 1
        sp = spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))
        print file,
        print "%15s" % str(total_size), "%15s" % str(not_found),
        print "%15.4f" % sp
        print ''


class AutoEncoder(nn.Module):
    def __init__(self, args, model_para):
        super(AutoEncoder, self).__init__()
        self.tdim = args.text_dim
        self.tdim1 = args.text_dim1
        self.tdim2 = args.text_dim2
        self.idim = args.image_dim
        self.idim1 = args.image_dim1
        self.idim2 = args.image_dim2
        self.sdim = args.sound_dim
        self.sdim1 = args.sound_dim1
        self.sdim2 = args.sound_dim2
        self.zdim = args.multi_dim
        self.brain_dim1 = args.brain_dim1
        self.brain_dim = args.brain_dim
        self.model_para = model_para

        # vector modality
        self.text_weight = nn.Sequential(
            nn.Linear(self.tdim, self.tdim),
            nn.Tanh()
        )
        self.image_weight = nn.Sequential(
            nn.Linear(self.idim, self.idim),
            nn.Tanh()
        )
        self.sound_weight = nn.Sequential(
            nn.Linear(self.idim, self.idim),
            nn.Tanh()
        )
        # value modality
        # self.text_weight = nn.Sequential(
        #     nn.Linear(self.tdim, 1),
        #     nn.Tanh()
        # )
        # self.image_weight = nn.Sequential(
        #     nn.Linear(self.idim, 1),
        #     nn.Tanh()
        # )
        # self.sound_weight = nn.Sequential(
        #     nn.Linear(self.idim, 1),
        #     nn.Tanh()
        # )

        self.encoder1 = nn.Sequential(
            nn.Linear(self.tdim, self.tdim1),
            nn.Tanh(),
            nn.Linear(self.tdim1, self.tdim2),
            nn.Tanh()
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(self.idim, self.idim1),
            nn.Tanh(),
            nn.Linear(self.idim1, self.idim2),
            nn.Tanh()
        )
        self.encoder3 = nn.Sequential(
            nn.Linear(self.sdim, self.sdim1),
            nn.Tanh(),
            nn.Linear(self.sdim1, self.sdim2),
            nn.Tanh()
        )
        self.encoder4 = nn.Sequential(
            nn.Linear(self.tdim2 + self.idim2 + self.sdim2, self.zdim),
            nn.Tanh()
        )

        self.decoder4 = nn.Sequential(
            nn.Linear(self.zdim, self.tdim2 + self.idim2 + self.sdim2),
            nn.Tanh()
        )

        self.decoder3 = nn.Sequential(
            nn.Linear(self.tdim2, self.tdim1),
            nn.Tanh(),
            nn.Linear(self.tdim1, self.tdim),
            nn.Tanh()
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(self.idim2, self.idim1),
            nn.Tanh(),
            nn.Linear(self.idim1, self.idim),
            nn.Tanh()
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(self.sdim2, self.sdim1),
            nn.Tanh(),
            nn.Linear(self.sdim1, self.sdim),
            nn.Tanh()
        )
        self.decoder_brain = nn.Sequential(
            nn.Linear(self.zdim, self.brain_dim1),
            nn.Tanh(),
            nn.Linear(self.brain_dim1, self.brain_dim),
            nn.Sigmoid()
        )

        self.reset_parameters()
        self.load_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.text_weight[0].weight.data)
        init.kaiming_normal(self.image_weight[0].weight.data)
        init.kaiming_normal(self.sound_weight[0].weight.data)
        init.constant(self.text_weight[0].bias.data, val=0)
        init.constant(self.image_weight[0].bias.data, val=0)
        init.constant(self.sound_weight[0].bias.data, val=0)
        init.kaiming_normal(self.decoder_brain[0].weight.data)
        init.kaiming_normal(self.decoder_brain[2].weight.data)
        init.constant(self.decoder_brain[0].bias.data, val=0)
        init.constant(self.decoder_brain[2].bias.data, val=0)

    def load_parameters(self):

        self.encoder1[0].weight.data = copy.deepcopy(self.model_para['encoder1.0.weight'])
        self.encoder1[2].weight.data = copy.deepcopy(self.model_para['encoder1.2.weight'])
        self.encoder1[0].bias.data = copy.deepcopy(self.model_para['encoder1.0.bias'])
        self.encoder1[2].bias.data = copy.deepcopy(self.model_para['encoder1.2.bias'])

        self.encoder2[0].weight.data = copy.deepcopy(self.model_para['encoder2.0.weight'])
        self.encoder2[2].weight.data = copy.deepcopy(self.model_para['encoder2.2.weight'])
        self.encoder2[0].bias.data = copy.deepcopy(self.model_para['encoder2.0.bias'])
        self.encoder2[2].bias.data = copy.deepcopy(self.model_para['encoder2.2.bias'])

        self.encoder3[0].weight.data = copy.deepcopy(self.model_para['encoder3.0.weight'])
        self.encoder3[2].weight.data = copy.deepcopy(self.model_para['encoder3.2.weight'])
        self.encoder3[0].bias.data = copy.deepcopy(self.model_para['encoder3.0.bias'])
        self.encoder3[2].bias.data = copy.deepcopy(self.model_para['encoder3.2.bias'])

        self.encoder4[0].weight.data = copy.deepcopy(self.model_para['encoder4.0.weight'])
        self.encoder4[0].bias.data = copy.deepcopy(self.model_para['encoder4.0.bias'])

        self.decoder1[0].weight.data = copy.deepcopy(self.model_para['decoder1.0.weight'])
        self.decoder1[2].weight.data = copy.deepcopy(self.model_para['decoder1.2.weight'])
        self.decoder1[0].bias.data = copy.deepcopy(self.model_para['decoder1.0.bias'])
        self.decoder1[2].bias.data = copy.deepcopy(self.model_para['decoder1.2.bias'])

        self.decoder2[0].weight.data = copy.deepcopy(self.model_para['decoder2.0.weight'])
        self.decoder2[2].weight.data = copy.deepcopy(self.model_para['decoder2.2.weight'])
        self.decoder2[0].bias.data = copy.deepcopy(self.model_para['decoder2.0.bias'])
        self.decoder2[2].bias.data = copy.deepcopy(self.model_para['decoder2.2.bias'])

        self.decoder3[0].weight.data = copy.deepcopy(self.model_para['decoder3.0.weight'])
        self.decoder3[2].weight.data = copy.deepcopy(self.model_para['decoder3.2.weight'])
        self.decoder3[0].bias.data = copy.deepcopy(self.model_para['decoder3.0.bias'])
        self.decoder3[2].bias.data = copy.deepcopy(self.model_para['decoder3.2.bias'])

        self.decoder4[0].weight.data = copy.deepcopy(self.model_para['decoder4.0.weight'])
        self.decoder4[0].bias.data = copy.deepcopy(self.model_para['decoder4.0.bias'])

    def forward(self, x_t, x_i, x_s):
        mm0 = self.text_weight(x_t).expand_as(x_t) * 0.1 + 1
        mm1 = self.image_weight(x_i).expand_as(x_i) * 0.1 + 1
        mm2 = self.sound_weight(x_s).expand_as(x_s) * 0.1 + 1
        x_t = mm0 * x_t
        x_i = mm1 * x_i
        x_s = mm2 * x_s
        encoded_text = self.encoder1(x_t)
        encoded_image = self.encoder2(x_i)
        encoded_sound = self.encoder3(x_s)
        encoded_mid = self.encoder4(torch.cat((encoded_text, encoded_image, encoded_sound), dim=1))
        decoded_mid = self.decoder4(encoded_mid)
        decoded_text = self.decoder3(decoded_mid[:, 0:self.tdim2])
        decoded_image = self.decoder2(decoded_mid[:, self.tdim2:self.tdim2 + self.idim2])
        decoded_sound = self.decoder1(decoded_mid[:, self.tdim2 + self.idim2:])
        decoded_brain = self.decoder_brain(encoded_mid)
        return decoded_text, decoded_image, decoded_brain, decoded_sound, encoded_mid

if __name__ == '__main__':

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--total-data', required=True)
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--brain-data', required=True)
    parser.add_argument('--text-dim', required=True, type=int)
    parser.add_argument('--image-dim', required=True, type=int)
    parser.add_argument('--sound-dim', required=True, type=int)
    parser.add_argument('--text-dim1', required=True, type=int)
    parser.add_argument('--text-dim2', required=True, type=int)
    parser.add_argument('--image-dim1', required=True, type=int)
    parser.add_argument('--image-dim2', required=True, type=int)
    parser.add_argument('--sound-dim1', required=True, type=int)
    parser.add_argument('--sound-dim2', required=True, type=int)
    parser.add_argument('--brain-dim1', required=True, type=int)
    parser.add_argument('--brain-dim', required=True, type=int)
    parser.add_argument('--multi-dim', required=True, type=int)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--load-model', required=True)
    parser.add_argument('--outmodel', required=True)
    parser.add_argument('--regularization', default=-1, type=float)
    parser.add_argument('--gpu', default=-1, type=int)
    args = parser.parse_args()

    # total_data
    vocab = []
    total_text = []
    total_image = []
    total_sound = []
    num = 0
    for line in open(args.total_data):
        line = line.strip().split()
        total_text.append(np.array([float(i) for i in line[1:args.text_dim + 1]]))  # (9405, 300)
        total_image.append(np.array([float(i) for i in line[args.text_dim + 1:args.text_dim + args.image_dim + 1]]))  # (9405, 128)
        total_sound.append(np.array([float(i) for i in line[args.text_dim + args.image_dim + 1:]]))  # (9405, 128)
        vocab.append(line[0])
        num += 1
    total_text = torch.from_numpy(np.array(total_text)).type(torch.FloatTensor)
    total_image = torch.from_numpy(np.array(total_image)).type(torch.FloatTensor)
    total_sound = torch.from_numpy(np.array(total_sound)).type(torch.FloatTensor)

    # training dataset
    indata = open(args.train_data)  # 300*128
    text = []
    image = []
    sound = []
    for line in indata:
        line = line.strip().split()
        text.append(np.array([float(i) for i in line[1:args.text_dim + 1]]))  # (9405, 300)
        image.append(np.array([float(i) for i in line[args.text_dim + 1:args.text_dim + args.image_dim + 1]]))  # (9405, 128)
        sound.append(np.array([float(i) for i in line[args.text_dim + args.image_dim + 1:]]))  # (9405, 128)
    text = torch.from_numpy(np.array(text)).type(torch.FloatTensor)
    image = torch.from_numpy(np.array(image)).type(torch.FloatTensor)
    sound = torch.from_numpy(np.array(sound)).type(torch.FloatTensor)
    train_ind = range(len(image))

    indata = open(args.brain_data)  # 300*128
    brain_multi = []
    for line in indata:
        line = line.strip().split()
        brain_multi.append(np.array([float(i) for i in line[1:]]))
    brain_multi = torch.from_numpy(np.array(brain_multi)).type(torch.FloatTensor)

    model_para = torch.load(open(args.load_model, 'rb'))

    # Data Loader for easy mini-batch return in training
    if args.gpu > -1:
        train_loader = Data.DataLoader(dataset=train_ind, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        autoencoder = AutoEncoder(args, model_para).cuda(args.gpu)
    else:
        train_loader = Data.DataLoader(dataset=train_ind, batch_size=args.batch_size, shuffle=True)
        autoencoder = AutoEncoder(args, model_para)


    if args.gpu > -1:
        total_text = Variable(total_text.cuda(args.gpu))
        total_image = Variable(total_image.cuda(args.gpu))
        total_sound = Variable(total_sound.cuda(args.gpu))
    else:
        total_text = Variable(total_text)
        total_image = Variable(total_image)
        total_sound = Variable(total_sound)

    optimizer = torch.optim.Adam(autoencoder.decoder_brain.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()
    for ep in range(50):
        ep += 1
        for step, ind in enumerate(train_loader):
            if args.gpu > -1:
                batch_text = Variable(text[ind].view(-1, args.text_dim).cuda(args.gpu))  # batch x, shape (batch, 300)
                batch_image = Variable(image[ind].view(-1, args.image_dim).cuda(args.gpu))  # batch y, shape (batch, 128)
                batch_sound = Variable(
                    sound[ind].view(-1, args.sound_dim).cuda(args.gpu))  # batch y, shape (batch, 128)
                batch_brain = Variable(
                    brain_multi[ind].cuda(args.gpu))  # batch x, shape (batch, 300)
            else:
                batch_text = Variable(text[ind].view(-1, args.text_dim))  # batch x, shape (batch, 300)
                batch_image = Variable(image[ind].view(-1, args.image_dim))  # batch y, shape (batch, 128)
                batch_sound = Variable(sound[ind].view(-1, args.sound_dim))  # batch y, shape (batch, 128)
                batch_brain = Variable(brain_multi[ind])

            decoded_text, decoded_image, decoded_brain, decoded_sound,  _ = autoencoder(batch_text, batch_image, batch_sound)
            #loss = loss_func(decoded_text, batch_text)  + loss_func(decoded_image, batch_image) + loss_func(decoded_brain, batch_brain)    # mean square error
            loss = loss_func(decoded_brain, batch_brain)
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients

            if step % 100 == 0:
                print 'Epoch: ', ep, '| train loss: %.4f' % loss.data[0]

        #
        # if ep % 50 == 0:
        #     evaluation(ep, total_text, total_image, autoencoder, vocab, args)

    #fine-tune
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.1*args.lr)
    loss_func = nn.MSELoss()

    for ep in range(args.epoch):
        ep += 1
        for step, ind in enumerate(train_loader):
            if args.gpu > -1:
                batch_text = Variable(text[ind].view(-1, args.text_dim).cuda(args.gpu))  # batch x, shape (batch, 300)
                batch_image = Variable(
                    image[ind].view(-1, args.image_dim).cuda(args.gpu))  # batch y, shape (batch, 128)
                batch_sound = Variable(
                    sound[ind].view(-1, args.sound_dim).cuda(args.gpu))  # batch y, shape (batch, 128)
                batch_brain = Variable(
                    brain_multi[ind].cuda(args.gpu))  # batch x, shape (batch, 300)
            else:
                batch_text = Variable(text[ind].view(-1, args.text_dim))  # batch x, shape (batch, 300)
                batch_image = Variable(image[ind].view(-1, args.image_dim))  # batch y, shape (batch, 128)
                batch_sound = Variable(sound[ind].view(-1, args.sound_dim))  # batch y, shape (batch, 128)
                batch_brain = Variable(brain_multi[ind])

            decoded_text, decoded_image, decoded_brain, decoded_sound, _ = autoencoder(batch_text, batch_image,
                                                                                       batch_sound)
            # loss = loss_func(decoded_text, batch_text)  + loss_func(decoded_image, batch_image) + loss_func(decoded_brain, batch_brain)    # mean square error
            loss = loss_func(decoded_brain, batch_brain)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 100 == 0:
                print 'Epoch: ', ep, '| train loss: %.4f' % loss.data[0]

        if ep % 50 == 0:
            evaluation(ep, total_text, total_image, total_sound, autoencoder, vocab, args)