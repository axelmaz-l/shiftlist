# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:07:59 2020

@author: axel
"""


import unittest

import shiftlist
class DecalageTest(unittest.TestCase):
    def test_decalage(self):
        liste=[0,1,2]
        shiftlist.decalage(liste)
        self.assertEqual(liste[0],2)
        self.assertEqual(liste[1],0)
        self.assertEqual(liste[2],1)
        L_init=[]
        shiftlist.decalage(L_init)
        self.assertEqual(0,len(L_init))
        self.assertTrue(len(L_init)==0)
        
        L_init=3
        self.assertRaises(TypeError,shiftlist.decalage,L_init)
        
unittest.main()
    
