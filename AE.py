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
import cPickle as pickle
import argparse
from ranking import *

torch.manual_seed(1)    # reproducible

def evaluation(ep, text, image, sound, autoencoder, vocab, args):
    testfile = ['men-3k.txt', 'simlex-999.txt', 'semsim.txt', 'vissim.txt', 'simverb-3500.txt',
                'wordsim353.txt', 'wordrel353.txt', 'association.dev.txt', 'association.dev.b.txt']

    _, _, _, multi_rep = autoencoder(text, image, sound)
    word_vecs = multi_rep.data.cpu().numpy()
    torch.save(autoencoder.state_dict(), open(args.outmodel + '.parameters-' + str(ep), 'wb'))
    outfile = open(args.outmodel + '-' + str(ep)+ '.rep.txt', 'w')
    for ind, w in enumerate(word_vecs):
        outfile.write(vocab[ind] + ' ' + ' '.join([str(i) for i in w]) + '\n')

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
        #outfile1.write(testfile[ind]+'\t'+str(sp)+'\n')
    # r1, r2, r3 = eval_category(word_vecs)
    # outfile1.write('categorization'+'\t'+str(r1)+'\t'+str(r2)+'\t'+str(r3)+'\n')

class AutoEncoder(nn.Module):
    def __init__(self, args):
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
            nn.Linear(self.tdim2+self.idim2+self.sdim2, self.zdim),
            nn.Tanh()
        )

        self.decoder4 = nn.Sequential(
            nn.Linear(self.zdim, self.tdim2+self.idim2+self.sdim2),
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
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal(self.encoder1[0].weight.data)
        init.kaiming_normal(self.encoder1[2].weight.data)
        init.constant(self.encoder1[0].bias.data, val=0)
        init.constant(self.encoder1[2].bias.data, val=0)

        init.kaiming_normal(self.encoder2[0].weight.data)
        init.kaiming_normal(self.encoder2[2].weight.data)
        init.constant(self.encoder2[0].bias.data, val=0)
        init.constant(self.encoder2[2].bias.data, val=0)

        init.kaiming_normal(self.encoder3[0].weight.data)
        init.kaiming_normal(self.encoder3[2].weight.data)
        init.constant(self.encoder3[0].bias.data, val=0)
        init.constant(self.encoder3[2].bias.data, val=0)

        init.kaiming_normal(self.encoder4[0].weight.data)
        init.constant(self.encoder4[0].bias.data, val=0)


        init.kaiming_normal(self.decoder1[0].weight.data)
        init.kaiming_normal(self.decoder1[2].weight.data)
        init.constant(self.decoder1[0].bias.data, val=0)
        init.constant(self.decoder1[2].bias.data, val=0)

        init.kaiming_normal(self.decoder2[0].weight.data)
        init.kaiming_normal(self.decoder2[2].weight.data)
        init.constant(self.decoder2[0].bias.data, val=0)
        init.constant(self.decoder2[2].bias.data, val=0)

        init.kaiming_normal(self.decoder3[0].weight.data)
        init.kaiming_normal(self.decoder3[2].weight.data)
        init.constant(self.decoder3[0].bias.data, val=0)
        init.constant(self.decoder3[2].bias.data, val=0)

        init.kaiming_normal(self.decoder4[0].weight.data)
        init.constant(self.decoder4[0].bias.data, val=0)


    def forward(self, x_t, x_i, x_s):
        encoded_text = self.encoder1(x_t)
        encoded_image = self.encoder2(x_i)
        encoded_sound = self.encoder3(x_s)
        encoded_mid = self.encoder4(torch.cat((encoded_text, encoded_image, encoded_sound), dim=1))
        decoded_mid = self.decoder4(encoded_mid)
        decoded_text = self.decoder3(decoded_mid[:,0:self.tdim2])
        decoded_image = self.decoder2(decoded_mid[:,self.tdim2:self.tdim2+self.idim2])
        decoded_sound = self.decoder1(decoded_mid[:,self.tdim2+self.idim2:])
        return decoded_text, decoded_image, decoded_sound, encoded_mid

if __name__ == '__main__':

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--train-data', required=True)
    parser.add_argument('--text-dim', required=True, type=int)
    parser.add_argument('--image-dim', required=True, type=int)
    parser.add_argument('--sound-dim', required=True, type=int)
    parser.add_argument('--text-dim1', required=True, type=int)
    parser.add_argument('--text-dim2', required=True, type=int)
    parser.add_argument('--image-dim1', required=True, type=int)
    parser.add_argument('--image-dim2', required=True, type=int)
    parser.add_argument('--sound-dim1', required=True, type=int)
    parser.add_argument('--sound-dim2', required=True, type=int)
    parser.add_argument('--multi-dim', required=True, type=int)
    parser.add_argument('--batch-size', required=True, type=int)
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--lr', default=0.005, type=float)
    parser.add_argument('--outmodel', required=True)
    parser.add_argument('--gpu', default=-1, type=int)
    args = parser.parse_args()

    # training dataset
    indata = open(args.train_data) #300*128
    vocab = []
    text = []
    image = []
    sound = []
    for line in indata:
        line = line.strip().split()
        vocab.append(line[0])
        text.append(np.array([float(i) for i in line[1:args.text_dim+1]]))  # (9405, 300)
        image.append(np.array([float(i) for i in line[args.text_dim+1:args.text_dim+args.image_dim+1]]))   # (9405, 128)
        sound.append(np.array([float(i) for i in line[args.text_dim+args.image_dim+1:]]))   # (9405, 128)
    text = torch.from_numpy(np.array(text)).type(torch.FloatTensor)
    image = torch.from_numpy(np.array(image)).type(torch.FloatTensor)
    sound = torch.from_numpy(np.array(sound)).type(torch.FloatTensor)
    train_ind = range(len(image))


    # Data Loader for easy mini-batch return in training
    if args.gpu > -1:
        train_loader = Data.DataLoader(dataset=train_ind, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        autoencoder = AutoEncoder(args).cuda(args.gpu)
    else:
        train_loader = Data.DataLoader(dataset=train_ind, batch_size=args.batch_size, shuffle=True)
        autoencoder = AutoEncoder(args)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    min_vloss = 99999
    if args.gpu > -1:
        total_text = Variable(text.cuda(args.gpu))
        total_image = Variable(image.cuda(args.gpu))
        total_sound = Variable(sound.cuda(args.gpu))
    else:
        total_text = Variable(text)
        total_image = Variable(image)
        total_sound = Variable(sound)

    for ep in range(args.epoch):
        ep += 1
        for step, ind in enumerate(train_loader):
            if args.gpu > -1:
                batch_text = Variable(text[ind].view(-1, args.text_dim).cuda(args.gpu))   # batch x, shape (batch, 300)
                batch_image = Variable(image[ind].view(-1, args.image_dim).cuda(args.gpu))   # batch y, shape (batch, 128)
                batch_sound = Variable(sound[ind].view(-1, args.sound_dim).cuda(args.gpu))  # batch y, shape (batch, 128)
            else:
                batch_text = Variable(text[ind].view(-1, args.text_dim))   # batch x, shape (batch, 300)
                batch_image = Variable(image[ind].view(-1, args.image_dim))   # batch y, shape (batch, 128)
                batch_sound = Variable(sound[ind].view(-1, args.sound_dim))  # batch y, shape (batch, 128)

            decoded_text, decoded_image, decoded_sound, _ = autoencoder(batch_text, batch_image, batch_sound)

            loss = loss_func(decoded_text, batch_text) + loss_func(decoded_image, batch_image) + loss_func(
                decoded_sound, batch_sound)  # mean square error
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if step % 100 == 0:
                print 'Epoch: ', ep, '| train loss: %.4f' % loss.data[0]

        if ep % 100 == 0:
            evaluation(ep, total_text, total_image, total_sound, autoencoder, vocab, args)

