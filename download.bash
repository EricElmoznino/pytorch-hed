#!/bin/bash

wget --verbose --continue --timestamping http://content.sniklaus.com/github/pytorch-hed/network-bsds500.pytorch
mv network-bsds500.pytorch network-bsds500.pth
