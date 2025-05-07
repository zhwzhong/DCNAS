# -*- coding: utf-8 -*-

from models.codo import Model

def make_model(args):
    return Model(args.scale)