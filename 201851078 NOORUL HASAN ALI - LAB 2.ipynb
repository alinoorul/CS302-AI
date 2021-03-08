{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# <center> IIIT Vadodara </center>\n",
    "## <center> Winter 2021-22 </center>\n",
    "## <center> CS 612, CS 652, CS/IT 308 Machine Learning </center>\n",
    "## <center> Lab#2 Bayes' classifier,k-means and k-NN</center>\n",
    "## <center> Noorul Hasan Ali (201851078)</center>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.stats import norm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Bayesian decision rule under normality assumption\n",
    "Hint: use classnote for same"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Consider the following data:\n",
    "Here, second column represent the class"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature|Class\n",
      "[[3.39 0.  ]\n",
      " [3.58 0.  ]\n",
      " [2.28 0.  ]\n",
      " [7.42 1.  ]\n",
      " [5.74 1.  ]\n",
      " [9.17 1.  ]\n",
      " [7.79 1.  ]\n",
      " [7.93 1.  ]\n",
      " [1.34 0.  ]\n",
      " [3.11 0.  ]]\n"
     ]
    }
   ],
   "source": [
    "dataset = np.array([[3.39,0],[3.58,0],[2.28,0],[7.42,1],[5.74,1],[9.17,1],[7.79,1],[7.93,1],[1.34,0],[3.11,0]])\n",
    "print('Feature|Class')\n",
    "print(dataset)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Find class probability $P_1$ and $P_2$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class Probability P1 = 0.5\n",
      "Class Probability P2 = 0.5\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x27e40cf8f48>]"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#write code here\n",
    "dataset_0=[]\n",
    "dataset_1=[]\n",
    "for i in dataset:\n",
    "    if(i[1]==0.0):\n",
    "        dataset_0.append(i[0])\n",
    "    else:\n",
    "        dataset_1.append(i[0])\n",
    "dataset_0=np.array(dataset_0)\n",
    "dataset_1=np.array(dataset_1)\n",
    "p1=len(dataset_0)/(len(dataset_0)+len(dataset_1))\n",
    "p2=len(dataset_1)/(len(dataset_0)+len(dataset_1))\n",
    "print(\"Class Probability P1 = {0}\".format(p1))\n",
    "print(\"Class Probability P2 = {0}\".format(p2))\n",
    "\n",
    "plt.plot(dataset_0,\"ro\")\n",
    "plt.plot(dataset_1,\"bo\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Find sample mean and sample standard deviation for each class "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [],
   "source": [
    "def mean(data):\n",
    "    return np.mean(data)\n",
    "def stddev(data):\n",
    "    return np.std(data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Plot $p_1(x)$ and $p_2(x)$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate the Gaussian probability distribution function for x\n",
    "def generate_distribution(data):\n",
    "    u=mean(data)\n",
    "    std=stddev(data)\n",
    "    dist=norm(u,std)\n",
    "    return dist\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x27e40ebf748>]"
      ]
     },
     "execution_count": 133,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXQAAAD6CAYAAACxrrxPAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/Il7ecAAAACXBIWXMAAAsTAAALEwEAmpwYAAAqgElEQVR4nO3deXhV9Z3H8fcXSABZRXADWZQACrIZl1EBE1wAEWm1iktdO+rj+Eyn1c7otGNbHR/HLradDu1Ia63Tat3aKu4iRMQNCAJKWExEZJElAgoCQpbf/PHNkey5uTn3/s499/t6njw3uffmnG8S+Nzf/Z3fIs45jDHGZL52vgswxhgTDgt0Y4yJCQt0Y4yJCQt0Y4yJCQt0Y4yJCQt0Y4yJiYQCXUQmicgaESkTkdsbefwaESkXkWU1H98Kv1RjjDHN6dDSE0SkPTATOAfYCCwWkdnOuZX1nvq4c+6WRE/cu3dvN3DgwNbUaowxWW/JkiWfOuf6NPZYi4EOnAKUOefWAojIY8CFQP1Ab5WBAwdSXFzclkMYY0zWEZGPm3oskS6XvsCGWl9vrLmvvotE5D0ReUpEjmmikBtEpFhEisvLyxM4tTHGmESFdVH0WWCgc24kMAd4uLEnOedmOefynXP5ffo0+o7BGGNMkhIJ9E1A7RZ3v5r7vuKc2+6c21/z5e+Bk8IpzxhjTKISCfTFQJ6IDBKRXGAGMLv2E0TkqFpfTgNWhVeiMcaYRLR4UdQ5VykitwAvA+2BPzjnSkTkLqDYOTcb+GcRmQZUAjuAa1JYszHGmEaIr+Vz8/PznY1yMcaY1hGRJc65/MYes5mixhgTE4mMQzcmdZyDv/wFdu2CvDyYMAE62D9LY5Jh/3OMX7/6FXznOwe/vuIK+POf/dVjTAazLhfjz2uvwW23wfTpsH493HorPPIIPPec78qMyUh2UdT4sXs3HHccHHYYLFwI3bvDgQOQnw87dkBJCfTo4btKYyLHLoqa6HnySSgvh9/9TsMcIDcXHnwQNm+Ge+/1W58xGcgC3fjx4IMwbBiccUbd+08+GaZO1X706mo/tRmToSzQTfqtXg1vvQXXXQciDR+/7DLYtAkWLEh/bcZkMAt0k34PPQTt28NVVzX++AUXQJcu8Oij6a3LmAxngW7Sq7ISHn5Yu1WOOKLx53TpAhdeCE89pRdKjTEJsUA36bVwIWzdClde2fzzLr9cR7u88kp66jImBizQTXrNnav95oWFzT/v3HPh0EO1lW6MSYgFukmvuXNhzBjo1av55+XkQEEBzJ+fnrqMiQELdJM+e/fC22/DxImJPX/CBFi3TmeRGmNaZIFu0ueNN6CionWBDtZKNyZBFugmfebO1a6UM89M7PkjRkDPnhboxiTIAt2kz9y5cNppOiwxEe3bw7hxFujGJMgC3aTHzp3w7rstj26pb8IEKCuDTz5JTV3GxIgFukmPRYt0M4tx41r3fdaPbkzCLNBNegRLJZ90Uuu+b/Ro6NbNAt2YBFigm/QoLtYt5nr2bN33deigKzDa2vnGtMgC3aRHcbFuXpGMk06C99+3dV2MaYEFukm9LVtg40ZtaSdj7FgN85Urw63LmJixQDept2SJ3ibbQh87Vm/ffTeceoyJKQt0k3rFxbog15gxyX3/4MHQtasFujEtsEA3qVdcDMcfr6GcjHbt9MXAAt2YZlmgm9Ryrm0XRAMnnQTLlkFVVShlGRNHFugmtT75RC+KJntBNDB2LOzbp/uRGmMaZYFuUmv5cr0dPbptx7ELo8a0yALdpNb77+vtiBFtO87QodC5swW6Mc2wQDeptWIF9OvX+hmi9XXoACNHaj+6MaZRFugmtVasgBNPDOdYw4dDSUk4xzImhhIKdBGZJCJrRKRMRG5v5nkXiYgTkTYOaTCxUFkJq1a1vbslMHw4lJfrhzGmgRYDXUTaAzOBycAJwGUickIjz+sGfBtYGHaRJkN9+CHs3x9uoIO10o1pQiIt9FOAMufcWufcAeAx4MJGnnc3cB/wZYj1mUwW1gXRgAW6Mc1KJND7Ahtqfb2x5r6viMhY4Bjn3PPNHUhEbhCRYhEpLre3zfG3YoVO+T/++HCO17cvdO9ugW5ME9p8UVRE2gH3A7e29Fzn3CznXL5zLr9Pnz5tPbWJuhUrdB2Wzp3DOZ6ItvYt0I1pVCKBvgk4ptbX/WruC3QDRgCvicg64DRgtl0YNaGOcAkEI12cC/e4xsRAIoG+GMgTkUEikgvMAGYHDzrnPnfO9XbODXTODQTeAaY552yLmWy2bx+UlobXfx4YPhy2b4dt28I9rjEx0GKgO+cqgVuAl4FVwBPOuRIRuUtEpqW6QJOhVq+G6uqDFzLDYhdGjWlSh0Se5Jx7AXih3n13NvHcs9pelsl4q1bpbSoDvbAw3GMbk+FspqhJjTVrdB3zwYPDPe6RR8Khh1oL3ZhGWKCb1Fi9GgYNgo4dwz2uiLbSbX9RYxqwQDepsXo1DBuWmmMPHarvAIwxdVigm/BVV8MHH2jwpsLQoTrKZefO1BzfmAxlgW7Ct349fPllalvoYK10Y+qxQDfhC7aJS1WgB8e1QDemDgt0E74gaFPV5TJokG54YYFuTB0W6CZ8q1fr0MJUrdeTkwPHHWeBbkw9FugmfMEIF5HUncNGuhjTgAW6Cd+aNanrbgkMHQplZVBVldrzZJmKCpg3DxYu1M9NZrFAN+H6/HPYvDl1F0QDQ4fqbkgff5za82SJigq4/XZdcn7iRDjtNOjRA267TXcSNJnBAt2EK+gGSUegw8ERNSZpX34JF18M990H48fD3/8OTz4JF10EP/85TJliQ/4zRUKLcxmTsA8+0NshQ1J7ntpj0adMSe25YqyyEqZNgzlzYOZMuPnmg49dfDEUFMBNN8EFF8Brr+ngIhNd1kI34Sot1UW5jj02tefp3VtH0tiF0Tb5+c81zGfNqhvmgeuug4cegjffhHvuSX99pnUs0E24ysqgf//wF+WqT0S7dSzQk7ZyJdx5p7bE//Efm37eFVfAlVfCXXdpsJvoskA34SotDX/J3Kbk5en5TKtVV8O110K3btrV0pKZM2HAAO1+sYFF0WWBbsLjnAZsXl56zpeXB5s2wd696TlfjPz1r7BoEdx/Pxx+eMvP794d7r1Xt4l9/PHU12eSY4FuwrN9O3z2WXoDHbSbxySsuhr+8z/1uvIVVyT+fd/4BowcCT/8oQ1ljCoLdBOeIFjTHejW7dIqs2fDe+/BD34A7dsn/n3t2sHdd+uf+eGHU1efSZ4FuglPEKzp7EMHa6G3gnN6cXPwYJgxo/Xff8EFcPLJ8F//pS19Ey0W6CY86RqyGOjWDY44wlrorfD667B0qc4KTWZMuQh8+9v6Gjp3bvj1mbaxQDfhKS3VoRC5uek75+DBFuitMGsW9OwJl1+e/DEuvlinAfzmN6GVZUJigW7Ck84RLgEbupiwTz+Fp56Cb34TOndO/jgdO8K3vqV98Rs2hFefaTsLdBMO5/R9uI9A37wZvvgivefNQP/3f3DgANxwQ9uPdeON+iefNavtxzLhsUA34fj0U11pMV0XRAN2YTQhQfiefjqMGNH24w0cCJMn67IAdnE0OizQTTiCbg8fLfTa5zeNeucdXSXhW98K75hXXqnzuhYsCO+Ypm0s0E04fAV68I7AWujNevxx7fu+6KLwjjltGhxyCDz6aHjHNG1jgW7CUVams1QGDkzvebt2hSOPtBZ6M6qq4IkntIuke/fwjtulC0yfrhdaDxwI77gmeRboJhw+hiwGbKRLs954Q68bJzORqCWXXw47dsArr4R/bNN6FugmHD6GLAYs0Jv1+OPaNTJ1avjHPvdcOOwweOSR8I9tWs8C3bRduldZrC8vD7ZuhV27/Jw/wiortUtk6lTtIglbTg58/evw3HO6xavxywLdtF15Oeze7TfQwS6MNmLBAv3zXHJJ6s4xfbpOA5g3L3XnMIlJKNBFZJKIrBGRMhG5vZHHbxKR90VkmYi8ISInhF+qiax0L8pVnwV6k2bP1tEtkyal7hyFhXpt+umnU3cOk5gWA11E2gMzgcnACcBljQT2o865E51zo4GfAPeHXaiJMF9DFgPHHVe3DgNoT9gzz8DEianpbgl06qQvGLNn2yQj3xJpoZ8ClDnn1jrnDgCPARfWfoJzrnbnZRfAhVeiibzSUj9DFgNdusDRR1ug17NyJXz0kY4XT7Xp02HLFt0FyfiTSKD3BWovwbOx5r46ROSfRORDtIX+z+GUZzJCWRkMGqRXyHyxkS4NzJ6tt6kY3VLflCn6mv7MM6k/l2laaBdFnXMznXPHAf8G/KCx54jIDSJSLCLF5eXlYZ3a+JbOjaGbYoHewOzZkJ8PfRs0v8J36KEwYYIFum+JBPom4JhaX/erua8pjwHTG3vAOTfLOZfvnMvv06dPwkWaCPM9ZDGQl6fDOT7/3G8dEbF1KyxcmJ7ulsDUqbBqFaxbl75zmroSCfTFQJ6IDBKRXGAGMLv2E0Sk9v/m8wFrKmWLrVt1zFoUAh1spEuNl17S19p0dLcEJk8+eG7jR4uB7pyrBG4BXgZWAU8450pE5C4RCV7/bxGREhFZBnwXuDpVBZuISffG0E2xVRfrePll3Z1v1Kj0nXPoUF394cUX03dOU1dCuwo6514AXqh33521Pv92yHWZTOF7DHrAhi5+papK11Y5/3zd4jVdRLSV/qc/6azRjh3Td26jbKaoaZvSUt1t2NeQxUDnztCvnwU6sGQJbN+e2slETZk8GfbsgTffTP+5jQW6aavSUh2ymMwW8mGzkS6A9mGLwDnnpP/chYU6etW6XfywQDdtE4URLgELdED7z/PzoXfv9J+7a1cYN84C3RcLdJM8XxtDNyUvT/sadu70XYk3O3fqdnM+ulsC550HJSW6BrtJLwt0k7zNm7XDNEqBDlk9dHHePF1P5dxz/dUwceLBWkx6WaCb5PlelKs+G7pIUZEubXPqqf5qGD1aZ47OneuvhmxlgW6SF5Ux6IFjj9WrgVkc6PPmaR+2z2V12reHggINdGfL9KWVBbpJXmmpJscxx7T83HTo1ElrydJA37JFp94XFPiuRLtd1q+HDz/0XUl2sUA3ySst1VZxFIYsBrJ4pMtrr+ltVAIdrNsl3SzQTfKiNGQxkJeXtRdF582DHj1gzBjflcCQIbrKowV6elmgm+RUV0dryGIgLw927NCPLFNUBOPHR+MNk4i20oNRNyY9LNBNcj75BPbti2agQ9Z1u2zYoK+vUehuCUycqNMC3nvPdyXZwwLdJCdqQxYDWRroRUV6W1jot47arB89/SzQTXKiGuiDBukSg1kY6IcdBiee6LuSg/r21SV1LdDTxwLdJKe0VNdHjcqQxUDHjtC/f1YFunPaV33WWeldLjcRhYXw+utw4IDvSrJDxP78JmOUluoa5FFLEMi6oYsffaRjvqPUfx6YOFFXh1i0yHcl2SGC/xtNRojikMVAEOhZMk0xWDMlioFeUKAjXqzbJT0s0E3rVVfrFEDfuxQ1JS9PN4vevt13JWlRVKTbzR1/vO9KGurVS8fFW6CnhwW6ab2NG3WPsSi30CErul2c00APWsJRNHGiLum7Z4/vSuLPAt20XlRHuASyKNDXrNFVjKPY3RIoLISKCnjrLd+VxJ8Fumm9qAf6wIG65F8WBHoUx5/Xd+aZOns1qNWkjgW6ab3SUl3ZsG9f35U0LjcXBgzImkDv108HHEVV165w8sm24UU6WKCb1ist1QuiURyyGMiCRbqqqzXQCwuj238eKCyE4mLYvdt3JfEW4f+RJrKiPGQxkAVDF0tK4NNPo91/HigogKoqWLDAdyXxZoFuWqeqCtauzYxA37ULyst9V5IyUR5/Xt/pp2tPmPWjp5YFummd9et1HncmBDrEuh+9qEj3FxkwwHclLevcGU47zQI91SzQTesEARnVSUWBmAd6VRXMn58ZrfNAQQG8+y7s3Om7kviyQDetE7WNoZsyYECshy4uWwaffZZZgV5YqJc0Xn/ddyXxZYFuWmfNGujSBY4+2nclzcvJ0aV0YzrSJei6yKRAP/VUHe1q3S6pY4FuWmf1ahg2LPrj5CDWqy4WFela41F/Xa2tY0c44wwL9FSyQDets2aNJkkmiOnQxYoK7baI8uzQphQU6JZ0n37qu5J4skA3idu7Fz7+WFvomWDYMPjiC93/NEaWLNEfK5O6WwJBzfPn+60jrizQTeI++EBvM6WFHrzwrF7tt46QBePPzzrLaxlJOflkvQRjywCkRkKBLiKTRGSNiJSJyO2NPP5dEVkpIu+JyFwRyYCRsabV1qzR20xqoQOsWuW3jpAVFeneoX36+K6k9XJyYNw460dPlRYDXUTaAzOBycAJwGUickK9py0F8p1zI4GngJ+EXaiJgNWr9WJo1IcsBo48Erp3j1ULff9+ePPNzOxuCRQU6Gvsli2+K4mfRFropwBlzrm1zrkDwGPAhbWf4Jwrcs7trfnyHaBfuGWaSFi9Wsd3d+7su5LEiGgrPUaBvnAh7NuX+YEO8NprXsuIpUQCvS+wodbXG2vua8r1wIuNPSAiN4hIsYgUl8d4jY3YWrMmc7pbAscfH6tALyrS16kJE3xXkrwxY/SNk3W7hC/Ui6IiciWQD/y0scedc7Occ/nOufw+mdgBmM2qqzNryGJg2DDYtCk267bOm6eBeOihvitJXocOMH68BXoqJBLom4Bjan3dr+a+OkTkbOD7wDTn3P5wyjORsWmTDlvMtBZ6UG9wQTeD7dune3Nm4vjz+goLdYrAxo2+K4mXRAJ9MZAnIoNEJBeYAcyu/QQRGQM8gIb5tvDLNN4F3RaZ2EKHWHS7vPWWLnSZyf3ngeBnsFZ6uFoMdOdcJXAL8DKwCnjCOVciIneJyLSap/0U6Ao8KSLLRGR2E4czmSoIxExroR93nL7Hj8HQxXnzdL2xceN8V9J2I0dCr14W6GHrkMiTnHMvAC/Uu+/OWp+fHXJdJmpWrdIrWUce6buS1snJ0VCPQQu9qEgn5nTr5ruStmvXTi/sWqCHy2aKmsSUlMDw4ZmxKFd9MRi6uHs3LF4cj+6WQEEBrFsHH33ku5L4sEA3LXPuYKBnomHD9ApcRYXvSpL2xhtQWRm/QAdrpYfJAt20rLwctm/P3EAfPlzDPIPXRp83T3uPzjjDdyXhGT5cly+wQA+PBbppWUmJ3mZqoI8YobcrVvitow3mzNEwP+QQ35WER0Rb6UVFsVvh2BsLdNOyINBPqL+ET4YYNkyvwgU/R4bZuhWWL4dzzvFdSfgKCnSKQwa/eYoUC3TTspIS6NEjs7bHqa1zZx3pkqEt9Llz9TaugQ7W7RIWC3TTspUrM3eES2D48Ixtoc+Zo1P9x471XUn4hgyBo46yQA+LBbppXqaPcAmMGKEjXfZn1qoUzmmgT5yok4riRkSXAbB+9HBYoJvmbduW2SNcAsOHQ1VVxq3psnq19jHHsbslUFCg1wlWrvRdSeazQDfNy/QRLoEMHekyZ47exjnQg5/t5Zf91hEHFuimeUGzKVNHuASGDNE1XTKsH33OHL2eO2iQ70pSp39/XbbeAr3tLNBN895/X6/IHXWU70raJjdXQz2DWugVFbqrT5xb54FJk2D+fF2h2STPAt00b/lyGDUqs0e4BDJspMs778AXX2RPoO/fr6FukmeBbppWVaUt9FGjfFcSjhEjYO1a2LPHdyUJmTNH50PFYUOLlowbB506WbdLW1mgm6Z9+KG+B45LoI8apWPj3nvPdyUJmTNHl8vt2dN3JanXuTOcdRa89JLvSjKbBbpp2rJlejt6tM8qwjNmjN4GP1eEffYZLFqUHd0tgUmTdFTpunW+K8lcFuimacuX68iQTB/hEjjmGL3AmwGBXlSk+3JnW6CDdbu0hQW6adry5bqwVceOvisJh4i+21i61HclLXrlFejSBU47zXcl6TNkCAwYYN0ubWGBbpoWjHCJkzFj9EJvZaXvSprkHLzwApx9to62zBYi2kqfOzej9yLxygLdNG77dti4MX6BPno0fPllpJcAKCmB9evh/PN9V5J+kybpdntvv+27ksxkgW4at3y53sblgmgg+Hki3I/+Qs127JMn+63Dh8JCvWxj3S7JsUA3jQsCPW4t9OCaQIQD/fnn9dfer5/vStKve3c4/XQL9GRZoJvGLV0KRx4Jhx/uu5Jw5eToBKOIXhjduRPefDM7u1sCkybpn2frVt+VZB4LdNO4RYvglFN8V5Eao0drCz2CC3C/8opO0J0yxXcl/gRdTUHXk0mcBbpp6PPP9aLhySf7riQ1xozRi77r1/uupIHnn4devbJruGJ9o0bpCozPPOO7ksxjgW4aWrJEb+Ma6ME7j0WL/NZRT0UFPPccTJ0az92JEiUC06bpuxVbfbF1LNBNQ4sX621+vt86UmXUKL0wunCh70rqWLBA+9CnT/ddiX/TpsG+ffDqq74rySwW6KahRYt0V4XDDvNdSWrk5mq3S8Ra6E8/rSsOnnuu70r8mzBBR7xYt0vrWKCbhhYvjm93S+CUU7RrKSIzRp3TQD/3XJ3yn+1yc/XC8LPP6kVikxgLdFPXli2wYUP8A/3UU7WDNiI7GC1dqr9262456MILobzcZo22hgW6qSvoP4/rkMXAqafqbUS6XZ5+WjezmDrVdyXRMWWKXup46inflWQOC3RT1+LFmizB2uFxdeyxeo0gAhdGndPQGjcO+vTxXU10dO8O552nv5vqat/VZIaEAl1EJonIGhEpE5HbG3l8vIi8KyKVInJx+GWatHn7bTjxxPh35Irou5AItNBXrIBVq+DSS31XEj2XXAKbNun+qqZlLQa6iLQHZgKTgROAy0Sk/o4H64FrgEfDLtCkUUUFvPWWNhWzwamn6tKGu3d7LePxx/VN0UUXeS0jki64QLtdnnjCdyWZIZEW+ilAmXNurXPuAPAYcGHtJzjn1jnn3gPsjVEmW7pULxSOH++7kvQ4/XTt73jrLW8lOKeBXlgYv2VzwtC9u67tYt0uiUkk0PsCG2p9vbHmvlYTkRtEpFhEisvLy5M5hEmlBQv0Nlta6Kefrmu1zp/vrYSlS6GszLpbmvONb2i3i412aVlaL4o652Y55/Kdc/l97OpP9Lz+OgwerKssZoMuXXQ27GuveSvh8cf1NeXrX/dWQuRNmwadO8Mjj/iuJPoSCfRNwDG1vu5Xc5+Jk+pqeOON7OluCUyYoCN79uxJ+6mrquDRR3UyUa9eaT99xujWDb72NX3x27/fdzXRlkigLwbyRGSQiOQCM4DZqS3LpN3KlbBjR/Z0twTOOktni3p4P19UpLv8XX112k+dcb75Tf3naUvqNq/FQHfOVQK3AC8Dq4AnnHMlInKXiEwDEJGTRWQj8A3gAREpSWXRJgWyrf88cMYZurShh370P/4RevbULgXTvLPP1p7AP/3JdyXR1iGRJznnXgBeqHffnbU+X4x2xZhMNX8+HHWUTrjJJt26wdixae9H37UL/vY3bZ136pTWU2ekDh3g8svh17/Wpezjum5cW9lMUaOdua++CuecoxNuss2ECTrBaN++tJ3yqaf0dNbdkrirrtKpEo/abJcmWaAbXXVw+3Yd8JuNCgvhwAEd5ZMmf/gDDBlycEkZ07JRo3RQ0qxZkdw9MBIs0I1usS6iLfRsNGGC9nu8+GJaTvfee7oR9I03Zucbora48UZdKsHjXLBIs0A38PLL2vTp3dt3JX4ccggUFKRtCMVvf6uvH9dck5bTxcqMGTp79IEHfFcSTRbo2W7nTl35KFu7WwLnnw+lpfqRQrt26UiNGTNs7HkyunaFK6/UtV22b/ddTfRYoGe7V1/VSUXZHuiTJ+ttilvpf/6zzmG6+eaUnibWbrxRJxg99JDvSqLHAj3bvfSSDoaO+4YWLTn2WBg2LKWBXl2tw+5OOin+G0Kl0siR2kP2q1/pqBdzkAV6Nqus1F14J0/Wgb7ZbsoUHY+eomUAnn0WVq+GW29NyeGzym236SxbW1a3Lgv0bFZUpB2Rl1ziu5JomDpVhy+mqJX+k5/AwIG6eqBpm0mT4IQT4Gc/syGMtVmgZ7MnntCrTOed57uSaBg/XueX/+UvoR/6jTd0qN2tt9qboTC0a6e/y2XLYO5c39VEhwV6tqqo0LnnwdqkRtd0ufRSbaF//nmoh77vPp2ufu21oR42q11xBRx9NNx1l7XSAxbo2WrePF2+zrpb6rrsMh1C8fe/h3bId96B556Df/mX+G/Vmk4dO8K//7uuK/fqq76riQZxnl7a8vPzXXFxsZdzG+D66+HJJ2HbNlsdqjbndJOPwYN1wlUIJk6E99+HtWu1h8uEZ/9+yMvTlvrbb2fHzFsRWeKcy2/sMWuhZ6Pdu7X//OtftzCvT0Rn/cydC1u3tvlwr76qb4a+/30L81To2BH+4z9g4UJ4/nnf1fhngZ6NHnkEvvgCbrrJdyXRdOWVugJlG2euVFfDHXdA//72q06la67RN1S33aaDlLKZBXq2cQ7+939h9Ghb6q8pxx+vKzD+5jc6Vj9JDz4IxcVwzz3akjSpkZMDv/wlrFkD//M/vqvxywI92yxcCMuXa5MxGzock3XLLbBhg84GSsL27XD77ToS8oorQq7NNHD++Tov7Mc/DqWnLGNZoGeb3/5Wd+m5/HLflUTbBRdoX8mvf53Ut99xh458nDnTXjfT5Re/0E1Dvvtd35X4Y4GeTT76SLd7ueYaDXXTtA4ddAWtoiJdwLwV5syB3/1OhymOGJGa8kxDQ4bAD36g/8T/9jff1fhhwxazyfXX6wXRtWt1nJdp3o4dOld/4sSEx6Xv2AEnngg9euhGUDZnK70qKuAf/gHWr9eNMA4/3HdF4bNhiwY+/BAefljXHrUwT0yvXjp04umndc/RFjinlya2bdNlci3M0y8nB/74R+3uuvZaHWmUTSzQs8Xdd+u/9ttv911JZvnOd6BPH52S2IJf/ELnat11F4wdm4baTKNGjID779cVHO6+23c16WWBng3eeENb57fcAkcd5buazNKtm4b53LnN7jn6yivwve/BRRfBv/1bGuszjbr5ZrjqKvjRj5IeqJSRrA897vbvhzFjdI3vkhKbrpiM/ft13P6ePTqHv0ePOg8vWwZnnaWDYt56y37FUbFvH5x5pq5BP2cOnH6674rCYX3o2ey++2DVKh2uaEmTnI4dtWN20yZthteyciWcc45uXPzss/YrjpLOnbXbpW9fHafeysFKGckCPc5ee007dGfM0FkXJnmnnqoXSH/3O5g9G4ClS3UATE6OrtcyYIDnGk0DRxyhrfMuXXTburff9l1Ralmgx9W6dXDxxTo494EHfFcTDz/+sW4IevnlvDTzQ8aP1zCfO1fXEjHRNGCALrHbq5e+AD/zjO+KUscCPY62btXt1Kqq9F9v9+6+K4qHTp2o+vts7mr/I86/ZSB5Aw/wzju69IuJtkGD4M03YfhwmD5dB3vFcYNpC/S4Wb9eFxBZuxb++lddLNqEYvVqKLjiaH646zYuz3mS+TtHcfTOEt9lmQQdfri21G+8US8tjRsXv351C/Q4mT9fL+Vv2aLj6AoLfVcUCzt2wL/+K4wcqYNcHn4Y/rRoGN2qPtNhFC+95LtEk6BOnXSx0cce0zbP2LE61WDbNt+VhcMCPQ5279b3kAUFcMghGuxnnum7qoz38cf6ax0wAH76U/jmN3WJ1quuQocxvv22DqGYPBluuAE++8xzxSZRl16q77iuuw7++7+1S+bWW3VCdUZzznn5OOmkk5xpox07nLv/fuf69HEOnLvuOud27/ZdVUbbts25Bx907rzznBPRjxkznHv//Sa+Yd8+5773PX1ijx7O/ehHzpWXp7Nk00Zr1jh3xRXOtW+v/40mTnTugQec27LFd2WNA4pdE7ma0MQiEZkE/ApoD/zeOfdf9R7vCPwfcBKwHbjUObeuuWPaxKIkbdum47Cee07XGPnyS22Z33uvbVjRSvv3ayttxQp49119Y7N0qa7/MWAAXH21tuASGo64bJmOgnn6aV2pccoU+NrXdFjFMcek+CcxYfjkE/j973UdntJSvW/oUH2ze+aZcNppcOyxkJvrt87mJha1GOgi0h74ADgH2AgsBi5zzq2s9ZybgZHOuZtEZAbwNefcpc0d1wK9Fud076wvvtDZiHv2wK5dsHGjXuTcsEGXvl26VPsBAA47TN83Xn+9LRyCbiy0f7++vu3fr7/Czz8/+LF9u84LCj7WrdP/tMGGRLm5ukpfQYEuhT5mTJLrmK9YoZ3sjzwCmzfrff36wahROoS0Xz/tpunXD3r31i6yLl30tmNHWzw9ApzTi6UvvqgjY958E3bu1MfatdMFOAcP1nA/4ghd6ufww/XP2bWr/jmDP2mXLjrBKcw/a1sD/R+AHznnzqv5+g79od29tZ7zcs1z3haRDsAWoI9r5uDJBvofrl3Azx49Gjh4aEet35arc1NHnefVPEnva+JYwX1ODn5DQs9rUE7Tdei7vAbHq/O1CLTvgMvJhdwc6NhJbxEa+w2n+j4f52zqvspKDfGqqoaP1Sei//GOPlqn6Q8frkvdjhihWRtqy6u6Wq+gzpun6+guX65X4fbubfp72rU7GOzt2jX8aN++7tcimfECkOE1Vjth1f5jWfLlcEoPDKDsQH9KD/Tno4p+7KjqmdDhc6ggRyrIkUpypJKf/mMp1/w2uXfUzQV6hwS+vy+wodbXG4H6lXz1HOdcpYh8DhwGfFqvkBuAGwD69++fUPH19T46lxGH17skLdSNQ6lzU+fBhvcJgqv7tdRODqmpvV4hTZ2zkTsPHu/gg189T0TfonfogOToLTkdICcHCV7qc3O/+obG/t35uC8qdbRvryMXOnXSHAxuDzlEl1zp2VNve/WCI4/UiUBp0a6dtspHjTp4n3P6dmHjRv3YuVPfSuzde/B27159hXJOXxSqq/XVKvi8/kfUeVorqlVaqLEdMBwYThlQVuexiur2bD/QjfL93Snf3529lbnsqerEnsqO7K3q+NXtgeoOVFS3p8Lp7XGDD03Jj5JIoIfGOTcLmAXaQk/mGNPuOZVp94RaljHpIaKvMD172lZGMZEDHFnzEQWJDFvcBNS+qtOv5r5Gn1PT5dIDvThqjDEmTRIJ9MVAnogMEpFcYAYwu95zZgNX13x+MTCvuf5zY4wx4Wuxy6WmT/wW4GV02OIfnHMlInIXOh5yNvAg8CcRKQN2oKFvjDEmjRLqQ3fOvQC8UO++O2t9/iXwjXBLM8YY0xo29d8YY2LCAt0YY2LCAt0YY2LCAt0YY2IiocW5UnJikXLgYy8nb1xv6s1sjaCo1xj1+sBqDIvVGI5kahzgnOvT2APeAj1qRKS4qfURoiLqNUa9PrAaw2I1hiPsGq3LxRhjYsIC3RhjYsIC/aBZvgtIQNRrjHp9YDWGxWoMR6g1Wh+6McbEhLXQjTEmJizQjTEmJizQa4jIT0VktYi8JyJ/F5GevmsKiMgkEVkjImUicrvveuoTkWNEpEhEVopIiYh823dNTRGR9iKyVESe811LY0Skp4g8VfNvcVXNFpCRISLfqfkbrxCRv4hIJ981AYjIH0Rkm4isqHVfLxGZIyKlNbep2SYo+fpCzxwL9IPmACOccyPRTbHv8FwP8NUm3TOBycAJwGUicoLfqhqoBG51zp0AnAb8UwRrDHwbWOW7iGb8CnjJOTcMGEWEahWRvsA/A/nOuRHoctpRWSr7j8CkevfdDsx1zuUBc2u+9uWPNKwv9MyxQK/hnHvFOVezBzzvoDszRcEpQJlzbq1z7gDwGHCh55rqcM5tds69W/P5bjSE+vqtqiER6QecD/zedy2NEZEewHh0fwGccwecc595LaqhDkDnmp3JDgE+8VwPAM6519G9GGq7EHi45vOHgenprKm2xupLReZYoDfuOuBF30XUaGyT7siFZUBEBgJjgIWeS2nML4F/BaK6u/IgoBx4qKZb6Pci0sV3UQHn3CbgZ8B6YDPwuXPuFb9VNesI59zmms+3AEf4LKYFoWROVgW6iLxa0/dX/+PCWs/5PtqF8Ii/SjOTiHQF/gr8i3Nul+96ahORqcA259wS37U0owMwFvitc24MsAe/3QR11PRBX4i+8BwNdBGRK/1WlZiaLTEjOUY7zMxJaMeiuHDOnd3c4yJyDTAVmBihPVET2aTbOxHJQcP8Eefc33zX04gzgGkiMgXoBHQXkT8756IUSBuBjc654N3NU0Qo0IGzgY+cc+UAIvI34HTgz16ratpWETnKObdZRI4CtvkuqL6wMyerWujNEZFJ6Nvxac65vb7rqSWRTbq9EhFB+31XOefu911PY5xzdzjn+jnnBqK/w3kRC3Occ1uADSIytOauicBKjyXVtx44TUQOqfmbTyRCF20bUXvz+quBZzzW0kAqMsdmitao2eC6I7C95q53nHM3eSzpKzWtyl9ycJPue/xWVJeInAksAN7nYP/0v9fsRRs5InIWcJtzbqrnUhoQkdHoRdtcYC1wrXNup9eiahGRHwOXol0ES4FvOef2+60KROQvwFnocrRbgR8CTwNPAP3Rpbovcc7Vv3Dqs747CDlzLNCNMSYmrMvFGGNiwgLdGGNiwgLdGGNiwgLdGGNiwgLdGGNiwgLdGGNiwgLdGGNi4v8Blqa/1Xm+pX4AAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "x  = np.arange(-3,12,0.1)\n",
    "dist0=generate_distribution(dataset_0)\n",
    "dist1=generate_distribution(dataset_1)\n",
    "y0 = dist0.pdf(x)\n",
    "y1 = dist1.pdf(x)\n",
    "plt.plot(x,y0,\"r\",label='p1')\n",
    "plt.plot(x,y1,\"b\",label='p2')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Test for $x=3$  and $x=5$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class 1 (val:0)\n"
     ]
    }
   ],
   "source": [
    "x=3\n",
    "pA = p1 * dist0.pdf(x)\n",
    "pB = p2 * dist1.pdf(x)\n",
    "if(pA>pB):\n",
    "    print(\"Class 1 (val:0)\")\n",
    "else:\n",
    "    print(\"Class 2 (val:1)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class 2 (val:1)\n"
     ]
    }
   ],
   "source": [
    "x=5\n",
    "pA = p1 * dist0.pdf(x)\n",
    "pB = p2 * dist1.pdf(x)\n",
    "if(pA>pB):\n",
    "    print(\"Class 1 (val:0)\")\n",
    "else:\n",
    "    print(\"Class 2 (val:1)\")"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Homework:\n",
    "1. Try for this 2-D data using classnote\n",
    "![image.png](attachment:image.png)\n",
    "2. Implement Bayesian classifier for iris data.\n",
    "You can get help from following link. https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Using naive Bayes classifier for simplicity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Class 1 (val 0) with probability:99.98761599264799%\n",
      "Class 2 (val 1) with probability:99.99999999983898%\n"
     ]
    }
   ],
   "source": [
    "data=[{\"values\":np.array([3.39,3.11,1.34,3.58,2.28]), \"label\":0},\n",
    "        {\"values\":np.array([2.33,1.78,3.36,4.67,2.86]), \"label\":0},\n",
    "        {\"values\":np.array([7.42,5.74,9.17,7.79,7.93]), \"label\":1},\n",
    "        {\"values\":np.array([4.69,3.53,2.51,3.42,0.79]),\"label\":1}]\n",
    "dist10=generate_distribution([3.39,3.11,1.34,3.58,2.28])\n",
    "dist20=generate_distribution([3.39,3.11,1.34,3.58,2.28])\n",
    "dist0=[]\n",
    "dist1=[]\n",
    "p1=p2=0.5\n",
    "for i in data:\n",
    "    if(i[\"label\"]==0):\n",
    "        dist0.append(generate_distribution(i[\"values\"]))\n",
    "    else:\n",
    "        dist1.append(generate_distribution(i[\"values\"]))\n",
    "\n",
    "\n",
    "def probability(x,y):\n",
    "    pA = p1 * dist0[0].pdf(x) * dist0[1].pdf(y)\n",
    "    pB = p2 * dist1[0].pdf(x) * dist1[1].pdf(y)\n",
    "    pA = pA / (pA+pB)\n",
    "    pB = 1-pA\n",
    "    if(pA>pB):\n",
    "        print(\"Class 1 (val 0) with probability:{}%\".format(pA*100))\n",
    "    else:\n",
    "        print(\"Class 2 (val 1) with probability:{}%\".format(pB*100))\n",
    "    \n",
    "probability(3,2)\n",
    "probability(9.0,3.0)\n",
    "   "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##  2. IRIS DATA SET"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MEAN ACCURACY: 96.0\n",
      "ACCURACY OF FOLDS(5): [96.66666666666667, 96.66666666666667, 100.0, 93.33333333333333, 93.33333333333333]\n"
     ]
    }
   ],
   "source": [
    "from csv import reader\n",
    "import numpy as np\n",
    "from random import seed\n",
    "from random import randrange\n",
    "\n",
    "def load_csv(filename):\n",
    "    dataset = list()\n",
    "    with open(filename, 'r') as file:\n",
    "        csv_reader = reader(file)\n",
    "        for row in csv_reader:\n",
    "            if not row:\n",
    "                continue\n",
    "            dataset.append(row)\n",
    "    class_values=dict()\n",
    "    for i in dataset:\n",
    "        for j in range(len(i)-1):\n",
    "            i[j]=float(i[j].strip())\n",
    "        if(i[-1] not in class_values):\n",
    "            class_values[i[-1]]=len(class_values)\n",
    "        i[-1]=class_values[i[-1]]\n",
    "    return (dataset,class_values)\n",
    "def separate_classes(dataset):\n",
    "    separated=dict()\n",
    "    for row in dataset:\n",
    "        if(row[-1] not in separated):\n",
    "            separated[row[-1]]=list()\n",
    "        separated[row[-1]].append(row[0:4])\n",
    "    return separated\n",
    "def stats_dataset(dataset):\n",
    "    statistics=[(np.mean(column),np.std(column), len(column)) for column in zip(*dataset)]\n",
    "    return statistics\n",
    "def stats_all(dataset):\n",
    "    stats=dict()\n",
    "    for label in dataset:\n",
    "        stats[label]=stats_dataset(dataset[label])\n",
    "    return stats\n",
    "def gen_dist(mean,std):\n",
    "    dist=norm(mean,std)\n",
    "    return dist\n",
    "def dist(stats):\n",
    "    dists=dict()\n",
    "    for label in stats:\n",
    "        dists[label]=list()\n",
    "        for stat in stats[label]:\n",
    "            dists[label].append(gen_dist(stat[0],stat[1]))\n",
    "    return dists\n",
    "def class_len(dataset):\n",
    "    class_lengths=dict()\n",
    "    for label in dataset:\n",
    "        class_lengths[label]=len(dataset[label])\n",
    "    return class_lengths\n",
    "def prob(class_lengths):\n",
    "    total=0\n",
    "    for label in class_lengths:\n",
    "        total=total+class_lengths[label]\n",
    "    probs=dict()\n",
    "    for label in class_lengths:\n",
    "        probs[label]=class_lengths[label]/total\n",
    "    return probs\n",
    "\n",
    "def predict_class(row,dists):\n",
    "    prob=dict()\n",
    "    for label in dists:\n",
    "        prob[label]=class_prob[label]\n",
    "        for i in range(len(row)):\n",
    "            prob[label]=prob[label]*dists[label][i].pdf(row[i])\n",
    "    best_p=-1\n",
    "    best_val=None\n",
    "    for class_value,p in prob.items():\n",
    "        if best_val is None or best_p<p:\n",
    "            best_val=class_value\n",
    "            best_p=p\n",
    "    return best_val\n",
    "\n",
    "def split_data(dataset,n_folds):\n",
    "    data_copy=list(dataset)\n",
    "    data_folds=list()\n",
    "    fold_size=int(len(dataset)/n_folds)\n",
    "    for _ in range(n_folds):\n",
    "        fold=list()\n",
    "        while len(fold) < fold_size:\n",
    "            i=randrange(len(data_copy))\n",
    "            fold.append(data_copy.pop(i))\n",
    "        data_folds.append(fold)\n",
    "    return data_folds\n",
    "\n",
    "def evaluate(dataset,n_folds,dists):\n",
    "    data_fold=split_data(dataset,n_folds)\n",
    "    accuracy=list()\n",
    "    for fold in data_fold:\n",
    "        correct=0\n",
    "        for row in fold:\n",
    "            predicted = predict_class(row[0:4],dists)\n",
    "            actual = row[4]\n",
    "            if(actual==predicted):\n",
    "                correct=correct+1\n",
    "        accuracy.append((correct/len(fold))*100)\n",
    "    mean_accuracy=0\n",
    "    for i in accuracy:\n",
    "        mean_accuracy=mean_accuracy+i\n",
    "    mean_accuracy=mean_accuracy/len(accuracy)\n",
    "    return accuracy,mean_accuracy\n",
    "    \n",
    "seed(1)\n",
    "n_folds=5\n",
    "filename='iris.csv'\n",
    "data,class_values=load_csv(filename)\n",
    "dataset=separate_classes(data)\n",
    "stats=stats_all(dataset)\n",
    "dists=dist(stats)\n",
    "class_prob=prob(class_len(dataset))\n",
    "accuracy,mean=evaluate(data,n_folds,dists)\n",
    "\n",
    "print(\"MEAN ACCURACY: {}\".format(mean))\n",
    "print(\"ACCURACY OF FOLDS({}): {}\".format(n_folds,accuracy))\n",
    "\n"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. K-Nearest neighbour (KNN)\n",
    "\n",
    "Consider the following observations:\n",
    "![image.png](attachment:image.png)\n",
    "\n",
    "Use KNN technique to classify the test data using K=3.\n",
    "\n",
    "Hint: \n",
    "<br>\n",
    "1) Calculate the Euclidean distance between the new point and the existing points.\n",
    "<br>\n",
    "2) Sort out the points distance-wise.\n",
    "<br>\n",
    "3) Now select the k-neighbours. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature1|Feature2|Class\n",
      "[[-2. -1.  1.]\n",
      " [-2.  1.  2.]\n",
      " [-2.  2.  2.]\n",
      " [-1. -1.  1.]\n",
      " [ 1. -1.  1.]\n",
      " [ 1.  1.  3.]\n",
      " [ 1.  2.  3.]\n",
      " [ 2.  1.  3.]]\n"
     ]
    }
   ],
   "source": [
    "data = np.float32([[-2,-1,1],[-2,1,2],[-2,2,2],[-1,-1,1],[1,-1,1],[1,1,3],[1,2,3],[2,1,3]])\n",
    "print('Feature1|Feature2|Class')\n",
    "print(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted class: 1\n"
     ]
    }
   ],
   "source": [
    "from math import sqrt\n",
    "classes=set(data[0:,2])\n",
    "count=dict()\n",
    "for c in classes:\n",
    "    count[c]=0\n",
    "k=3\n",
    "def dist(p1,p2):\n",
    "    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5\n",
    "def test(testrow):\n",
    "    distarray=list()\n",
    "    for row in data:\n",
    "        distarray.append((dist(testrow,row[0:2]),int(row[2])))\n",
    "    distarray.sort(key= lambda e:e[0])\n",
    "    for i in range(k):\n",
    "        count[distarray[i][1]]+=1\n",
    "    max_val=-1\n",
    "    predict=None\n",
    "    for c,i in count.items():\n",
    "        if predict==None or i>max_val:\n",
    "            max_val=i\n",
    "            predict=c\n",
    "    print(\"Predicted class: {}\".format(int(predict)))\n",
    "\n",
    "test([1,-1])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. K-Means\n",
    "\n",
    "Ignore the class information of the training data given in problem 2 and use K-means algorithm to classify the same. Assume the initial starting classes as:  \n",
    "<br>\n",
    "C1={observations 1,2}\n",
    "<br>\n",
    "C2={observations 3,4,5}\n",
    "<br>\n",
    "C3={observations 6,7,8}\n",
    "<br>\n",
    "\n",
    "Now using the result find the class that would be assigned to the test data.\n",
    "\n",
    "<br>\n",
    "Hint: \n",
    "\n",
    "1) Find the centroid (mean) of each cluster based on the initial classes assignment given.\n",
    "\n",
    "2) Find out the Euclidean distance between each point and each cluster centroid. \n",
    "\n",
    "3) Assign all the points to the closest cluster centroid.\n",
    "\n",
    "4) Recompute centroids of newly formed clusters.\n",
    "\n",
    "5) Stop if the new centroids are same as old centroids else repeat step 2 to 4. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "centroid 1:[-2.          0.66666667]\n",
      "centroid 2:[ 0. -1.]\n",
      "centroid 3:[1.33333333 1.33333333]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import math\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import copy\n",
    "\n",
    "k=3\n",
    "initial_cluster = [1,1,2,2,2,3,3,3]\n",
    "\n",
    "cluster_data = data[:,:-1]\n",
    "index=0\n",
    "x=list()\n",
    "for row in cluster_data:\n",
    "    y=list()\n",
    "    for p in row:\n",
    "        y.append(p)\n",
    "    y=np.array(y)\n",
    "    x.append({\"point\":y,\"cluster\":initial_cluster[index]})\n",
    "    index+=1\n",
    "cluster_data=x\n",
    "\n",
    "def calc_distance(X1, X2):\n",
    "    return (sum((X1 - X2)**2))**0.5\n",
    "\n",
    "def calc_centroids(cluster_data):\n",
    "    centroids=dict()\n",
    "    for point in cluster_data:\n",
    "        if(point[\"cluster\"] not in centroids):\n",
    "            centroids[point[\"cluster\"]]=dict()\n",
    "            centroids[point[\"cluster\"]][\"point\"]=np.array([0.0,0.0])\n",
    "            centroids[point[\"cluster\"]][\"n\"]=0\n",
    "        centroids[point[\"cluster\"]][\"point\"]+=point[\"point\"]\n",
    "        centroids[point[\"cluster\"]][\"n\"]+=1\n",
    "        \n",
    "    for cluster in centroids:\n",
    "        centroids[cluster][\"point\"]/=centroids[cluster][\"n\"]\n",
    "        #centroid[\"point\"]/=centroid[\"n\"]\n",
    "    return centroids\n",
    "\n",
    "def calc_new_clusterlist(cluster_data,centroids):\n",
    "    new_cluster_data=copy.deepcopy(cluster_data)\n",
    "    for point in new_cluster_data:\n",
    "        min_dist=10.0\n",
    "        new_cluster=None\n",
    "        for centroid in centroids:\n",
    "            dist=calc_distance(centroids[centroid][\"point\"],point[\"point\"])\n",
    "            if dist<min_dist or new_cluster == None:\n",
    "                min_dist=dist\n",
    "                new_cluster=centroid\n",
    "        point[\"cluster\"]=new_cluster\n",
    "    return new_cluster_data\n",
    "\n",
    "centroids=calc_centroids(cluster_data)\n",
    "\n",
    "new_cluster_data=calc_new_clusterlist(cluster_data,centroids)\n",
    "\n",
    "\n",
    "def convert_cluster_data(cluster_data):\n",
    "    cluster_list=list()\n",
    "    for i in cluster_data:\n",
    "        cluster_list.append(i[\"cluster\"])\n",
    "    return cluster_list\n",
    "\n",
    "def check_cluster_change(old_cluster_data,new_cluster_data):\n",
    "    return not convert_cluster_data(old_cluster_data)==convert_cluster_data(new_cluster_data)\n",
    "\n",
    "centroids=calc_centroids(cluster_data)\n",
    "new_cluster_data=calc_new_clusterlist(cluster_data,centroids)\n",
    "iterations=20\n",
    "count=0\n",
    "while check_cluster_change(cluster_data,new_cluster_data) and count < iterations:\n",
    "    centroids=calc_centroids(new_cluster_data)\n",
    "    cluster_data=new_cluster_data\n",
    "    new_cluster_data=calc_new_clusterlist(cluster_data,centroids)\n",
    "\n",
    "for centroid in centroids:\n",
    "    print(\"centroid {}:{}\".format(centroid, centroids[centroid][\"point\"]))\n",
    "    plt.scatter(centroids[centroid][\"point\"][0],centroids[centroid][\"point\"][1],marker=\"^\",s=100,color='black')\n",
    "\n",
    "for point in cluster_data:\n",
    "    if(point[\"cluster\"]==1):\n",
    "        color=\"red\"\n",
    "    if(point[\"cluster\"]==2):\n",
    "        color=\"blue\"\n",
    "    if(point[\"cluster\"]==3):\n",
    "        color=\"green\"\n",
    "    plt.scatter(point[\"point\"][0],point[\"point\"][1],color=color,s=100)\n",
    "\n",
    "\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "name": "LAB 1 Introduction to python for CT314.ipynb",
   "provenance": [],
   "version": "0.3.2"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
