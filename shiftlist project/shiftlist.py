# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 17:55:35 2020

@author: axel
"""


def decalage(L):# décale vers la droite tous les éléments de la liste une fois
    
    if type(L)!=list:
        raise TypeError("Vous n'avez pas saisi de liste")
    long=len(L)
    if long==0:
        return
    val_courant=L[0]
    for i in range(1,long):
        val_prec=val_courant
        val_courant=L[i]
        L[i]=val_prec
    L[0]=val_courant
    


t=[]
u=decalage(t)
print(u)

    
    