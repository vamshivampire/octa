
import tkinter as tk


def cbc(op,text):
    if op=="1":
        return lambda : view_dataset(text)
    if op=="2":
        return lambda : describe_dataset(text)
    elif op=="3":
        return lambda : amount_and_transactions_graph()
    elif op=="3_1":
        return lambda : amount_and_transactions_text(text)
    elif op=="4":
        return lambda : transaction_and_time_hist()
    elif op=="4_1":
        return lambda : transaction_and_time_text(text)
    elif op=="5":
        return lambda : time_and_amount()

def func(op,X_train,Y_train,X_test,Y_test,text):  
    
    if op=='6':
       return lambda : gaussian_nb("s", X_train,Y_train,X_test,Y_test,text)
       
    elif op=='7':
       return lambda : random_forest("s",X_train,Y_train,X_test,Y_test,text)
      
         
    elif op=='8':
       return lambda : logistic_regression("s",X_train,Y_train,X_test,Y_test,text)
       
    
    elif op=='9':
       return lambda : svc("s",X_train,Y_train,X_test,Y_test,text)
       
                
    elif op=='10':
       return lambda : kneighbours("s",X_train,Y_train,X_test,Y_test,text)
       
    
    elif op=='11':
       return lambda : decisiontree("s",X_train,Y_train,X_test,Y_test,text)
      
       
    elif op=='12':
       return lambda : extratree("s",X_train,Y_train,X_test,Y_test,text)
       
    
    elif op=='13':
       return  lambda :adaboost("s",X_train,Y_train,X_test,Y_test,text)
      

    elif op=='14':
       return lambda :fc(text)


def fc(text):
        gaussian_nb("a",X_train,Y_train,X_test,Y_test,text)
        random_forest("a",X_train,Y_train,X_test,Y_test,text)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        logistic_regression("a",X_train,Y_train,X_test,Y_test,text)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        svc("a",X_train,Y_train,X_test,Y_test,text)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        kneighbours("a",X_train,Y_train,X_test,Y_test,text)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        decisiontree("a",X_train,Y_train,X_test,Y_test,text)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        extratree("a",X_train,Y_train,X_test,Y_test,text)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        adaboost("a",X_train,Y_train,X_test,Y_test,text)
        print("-------------------------------------------\n")
        print("-------------------------------------------")
        plt.show()    

    
def view_dataset(text):
    s=df
    text.delete(1.0,tk.END)
    text.insert(tk.END, s[:1000],"bold")
    text.update_idletasks()

def describe_dataset(text):
    s1="\n\n\n"+"DESCRIPTION OF DATASET\n"+"-------------------------\n"
    dataset_describe=df.describe()
    s=s1+str(dataset_describe)
    text.delete(1.0,tk.END)
    text.insert(tk.END, s[:1000],"bold")
    text.update_idletasks()

def amount_and_transactions_graph():
            f,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(10,10))
            bins = 20
            s1=df.Amount[df.Class==1]
            ax1.hist(s1[:20],bins=bins)
            ax1.set_title('Fraud')
            s2=df.Amount[df.Class==0]
            ax2.hist(s2[:20],bins=bins)
            ax2.set_title('Normal')
            plt.xlabel('Amount ($)')
            plt.ylabel('Number of Transactions')
            #plt.yscale('log')
            plt.show()
            
def amount_and_transactions_text(text):
            s1=df.Amount[df.Class==1].describe()
            s2=df.Amount[df.Class==0].describe()
            s = "\nFraud"+"\n-----\n"+str(s1)+"\nNormal"+"\n-----\n"+str(s2)
            text.delete(1.0,tk.END)
            text.insert(tk.END, s,"bold")
            text.see(tk.END)
            
def transaction_and_time_hist():
        f, (ax1,ax2) = plt.subplots(2,2,sharex=True,figsize=(12,))
        bins = 50
        ax1.hist(df.Time[df.Class==1],bins=bins)
        ax1.set_title('Fraud')
        ax2.hist(df.Time[df.Class==0],bins=bins)
        ax2.set_title('Normal')
        plt.xlabel('Time(in seconds)')
        plt.ylabel('Number of Transactions')
        plt.show()

def transaction_and_time_text(text):
        s1=df.Time[df.Class==1].describe()
        s2=df.Time[df.Class==0].describe()
        s="\nFraud"+"\n-----\n"+str(s1)+"\nNormal\n"+"\n-----\n\n"+str(s2)
        text.delete(1.0,tk.END)
        text.insert(tk.END, s,"bold")
        text.see(tk.END)
        
def time_and_amount():
    f,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(12,6))
    ax1.scatter(df.Time[df.Class==1],df.Amount[df.Class==1])
    ax1.set_title('Fraud')
    ax2.scatter(df.Time[df.Class==0],df.Amount[df.Class==0])
    ax2.set_title('Normal')
    plt.xlabel('Time (in seconds)')
    plt.ylabel('AMount in Dollars')
    plt.show()
###########################################
def roc_curve_acc(Y_test, Y_pred,method):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, color=next(colors),label='%s AUC = %0.3f'%(method, roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    #plt.show()
        
    
def gaussian_nb(typ,X_train,Y_train,X_test,Y_test,text):
    text.delete(1.0,tk.END)
    text.insert(tk.END, "\n\nIMPORTING GaussianNB"+"\nProcessing this might take a while...","bold")
    text.update_idletasks()
    from sklearn.naive_bayes import GaussianNB
    GNB=GaussianNB()
    text.insert(tk.END, "\n\n Number of Features for Training : "+str(len(X_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Labels for Training : "+str(len(Y_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n *** Training GaussianNB using the above Features and Labels ***","bold")
    text.update_idletasks()
    GNB.fit(X_train, Y_train)
    text.insert(tk.END, "\n\n Number of Test Features : "+str(len(X_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Predicting the Test Labels for the above Test Features using GaussianNB Classifier","bold")
    text.update_idletasks()
    Y_pred=GNB.predict(X_test)
    text.insert(tk.END, "\n\n Number of Actual Labels : "+str(len(Y_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Test Labels Predicted by GaussianNB --> "+str(len(Y_pred)),"bold")
    text.insert(tk.END, "\n\n ---------------------------------------------------")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of LABELS MATCHED : "+str(accuracy_score(Y_test,Y_pred,normalize=False)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Calculating Accuracy of GaussianNB = Label Matched/Actual Labels","bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Accuracy Score : "+str(accuracy_score(Y_test,Y_pred,normalize=True)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\nGaussian NB report \n"+classification_report(Y_test,Y_pred),"bold")
    text.update_idletasks()
    roc_curve_acc(Y_test, Y_pred,'GNB')
    if typ=="s":
       plt.show()
    elif typ=="a":
       pass 


def random_forest(typ,X_train,Y_train,X_test,Y_test,text):
    text.delete(1.0,tk.END)
    text.insert(tk.END, "\n\nIMPORTING RandomForest"+"\nProcessing this might take a while...","bold")
    text.update_idletasks()
    from sklearn.ensemble import RandomForestClassifier
    RF=RandomForestClassifier()
    text.insert(tk.END, "\n\n Number of Features for Training : "+str(len(X_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Labels for Training : "+str(len(Y_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n *** Training RandomForest using the above Features and Labels ***","bold")
    text.update_idletasks()
    RF.fit(X_train, Y_train)
    text.insert(tk.END, "\n\n Number of Test Features : "+str(len(X_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Predicting the Test Labels for the above Test Features using RandomForest Classifier","bold")
    text.update_idletasks()
    Y_pred=RF.predict(X_test)
    text.insert(tk.END, "\n\n Number of Actual Labels : "+str(len(Y_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Test Labels Predicted by RandomForest --> "+str(len(Y_pred)),"bold")
    text.insert(tk.END, "\n\n ---------------------------------------------------")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of LABELS MATCHED : "+str(accuracy_score(Y_test,Y_pred,normalize=False)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Calculating Accuracy of RandomForest = Label Matched/Actual Labels ","bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Accuracy Score : "+str(accuracy_score(Y_test,Y_pred,normalize=True)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n RandomForest report \n"+classification_report(Y_test,Y_pred),"bold")
    text.update_idletasks()
    roc_curve_acc(Y_test, Y_pred,'RF')
    if typ=="s":
       plt.show()
    elif typ=="a":
       pass 


def logistic_regression(typ,X_train,Y_train,X_test,Y_test,text):
    text.delete(1.0,tk.END)
    text.insert(tk.END, "\n\nIMPORTING Logistic Regression"+"\nProcessing this might take a while...","bold")
    text.update_idletasks()
    from sklearn.linear_model import LogisticRegression
    LR=LogisticRegression()
    text.insert(tk.END, "\n\n Number of Features for Training : "+str(len(X_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Labels for Training : "+str(len(Y_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n *** Training LogicalRegression using the above Features and Labels ***","bold")
    text.update_idletasks()
    LR.fit(X_train, Y_train)
    text.insert(tk.END, "\n\n Number of Test Features : "+str(len(X_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Predicting the Test Labels for the above Test Features using LogicalRegression ","bold")
    text.update_idletasks()
    Y_pred=LR.predict(X_test)
    text.insert(tk.END, "\n\n Number of Actual Labels : "+str(len(Y_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Test Labels Predicted by LogicalRegression --> "+str(len(Y_pred)),"bold")
    text.insert(tk.END, "\n\n ---------------------------------------------------")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of LABELS MATCHED : "+str(accuracy_score(Y_test,Y_pred,normalize=False)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Calculating Accuracy of LogicalRegression = Label Matched/Actual Labels ","bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Accuracy Score : "+str(accuracy_score(Y_test,Y_pred,normalize=True)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n LogicalRegression report \n"+classification_report(Y_test,Y_pred),"bold")
    text.update_idletasks()
    roc_curve_acc(Y_test, Y_pred,'LR')
    if typ=="s":
       plt.show()
    elif typ=="a":
       pass 



def svc(typ,X_train,Y_train,X_test,Y_test,text):
    text.delete(1.0,tk.END)
    text.insert(tk.END, "\n\nIMPORTING SVC"+"\nProcessing this might take a while...","bold")
    text.update_idletasks()
    from sklearn.svm import SVC
    SVM=SVC()
    text.insert(tk.END, "\n\n Number of Features for Training : "+str(len(X_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Labels for Training : "+str(len(Y_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n *** Training SVM using the above Features and Labels ***","bold")
    text.update_idletasks()
    SVM.fit(X_train, Y_train)
    text.insert(tk.END, "\n\n Number of Test Features : "+str(len(X_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Predicting the Test Labels for the above Test Features using SVM ","bold")
    text.update_idletasks()
    Y_pred=SVM.predict(X_test)
    text.insert(tk.END, "\n\n Number of Actual Labels : "+str(len(Y_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Test Labels Predicted by SVM --> "+str(len(Y_pred)),"bold")
    text.insert(tk.END, "\n\n ---------------------------------------------------")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of LABELS MATCHED : "+str(accuracy_score(Y_test,Y_pred,normalize=False)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Calculating Accuracy of SVM = Label Matched/Actual Labels ","bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Accuracy Score : "+str(accuracy_score(Y_test,Y_pred,normalize=True)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n SVM report \n"+classification_report(Y_test,Y_pred),"bold")
    text.update_idletasks()
    roc_curve_acc(Y_test, Y_pred,'SVM')
    if typ=="s":
       plt.show()
    elif typ=="a":
       pass 

       
    
def kneighbours(typ,X_train,Y_train,X_test,Y_test,text):
    text.delete(1.0,tk.END)
    text.insert(tk.END, "\n\nIMPORTING KNeighbours"+"\nProcessing this might take a while...","bold")
    text.update_idletasks()
    from sklearn.neighbors import KNeighborsClassifier
    KNN=KNeighborsClassifier()
    text.insert(tk.END, "\n\n Number of Features for Training : "+str(len(X_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Labels for Training : "+str(len(Y_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n *** Training KNeighbors using the above Features and Labels ***","bold")
    text.update_idletasks()
    KNN.fit(X_train, Y_train)
    text.insert(tk.END, "\n\n Number of Test Features : "+str(len(X_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Predicting the Test Labels for the above Test Features using KNeighbors ","bold")
    text.update_idletasks()
    Y_pred=KNN.predict(X_test)
    text.insert(tk.END, "\n\n Number of Actual Labels : "+str(len(Y_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Test Labels Predicted by KNeighbors --> "+str(len(Y_pred)),"bold")
    text.insert(tk.END, "\n\n ---------------------------------------------------")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of LABELS MATCHED : "+str(accuracy_score(Y_test,Y_pred,normalize=False)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Calculating Accuracy of nKNeighbors = Label Matched/Actual Labels ","bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Accuracy Score : "+str(accuracy_score(Y_test,Y_pred,normalize=True)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n KNeighbors report \n"+classification_report(Y_test,Y_pred),"bold")
    text.update_idletasks()
    roc_curve_acc(Y_test, Y_pred,'KNN')
    if typ=="s":
       plt.show()
    elif typ=="a":
       pass 
    


def decisiontree(typ,X_train,Y_train,X_test,Y_test,text):
    text.delete(1.0,tk.END)
    text.insert(tk.END, "\n\nIMPORTING DecisionTree"+"\nProcessing this might take a while...","bold")
    text.update_idletasks()
    from sklearn.tree import DecisionTreeClassifier
    DT=DecisionTreeClassifier()
    text.insert(tk.END, "\n\n Number of Features for Training : "+str(len(X_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Labels for Training : "+str(len(Y_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n *** Training Decision Tree using the above Features and Labels ***","bold")
    text.update_idletasks()
    DT.fit(X_train, Y_train)
    text.insert(tk.END, "\n\n Number of Test Features : "+str(len(X_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Predicting the Test Labels for the above Test Features using Decision Tree ","bold")
    text.update_idletasks()
    Y_pred=DT.predict(X_test)
    text.insert(tk.END, "\n\n Number of Actual Labels : "+str(len(Y_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Test Labels Predicted by Decision Tree --> "+str(len(Y_pred)),"bold")
    text.insert(tk.END, "\n\n ---------------------------------------------------")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of LABELS MATCHED : "+str(accuracy_score(Y_test,Y_pred,normalize=False)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Calculating Accuracy of Decision Tree = Label Matched/Actual Labels ","bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Accuracy Score : "+str(accuracy_score(Y_test,Y_pred,normalize=True)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Decision Tree report \n"+classification_report(Y_test,Y_pred),"bold")
    text.update_idletasks()
    roc_curve_acc(Y_test, Y_pred,'DT')
    if typ=="s":
       plt.show()
    elif typ=="a":
       pass 
    



def extratree(typ,X_train,Y_train,X_test,Y_test,text):
    text.delete(1.0,tk.END)
    text.insert(tk.END, "\n\nIMPORTING ExtraTree"+"\nProcessing this might take a while...","bold")
    text.update_idletasks()
    from sklearn.tree import ExtraTreeClassifier
    ETC=ExtraTreeClassifier()
    text.insert(tk.END, "\n\n Number of Features for Training : "+str(len(X_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Labels for Training : "+str(len(Y_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n *** Training ExtraTree using the above Features and Labels ***","bold")
    text.update_idletasks()
    ETC.fit(X_train, Y_train)
    text.insert(tk.END, "\n\n Number of Test Features : "+str(len(X_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Predicting the Test Labels for the above Test Features using ExtraTree ","bold")
    text.update_idletasks()
    Y_pred=ETC.predict(X_test)
    text.insert(tk.END, "\n\n Number of Actual Labels : "+str(len(Y_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Test Labels Predicted by DExtraTree --> "+str(len(Y_pred)),"bold")
    text.insert(tk.END, "\n\n ---------------------------------------------------")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of LABELS MATCHED : "+str(accuracy_score(Y_test,Y_pred,normalize=False)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Calculating Accuracy of ExtraTree = Label Matched/Actual Labels ","bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Accuracy Score : "+str(accuracy_score(Y_test,Y_pred,normalize=True)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n ExtraTree report \n"+classification_report(Y_test,Y_pred),"bold")
    text.update_idletasks()
    roc_curve_acc(Y_test, Y_pred,'ETC')
    if typ=="s":
       plt.show()
    elif typ=="a":
       pass 


    
def adaboost(typ,X_train,Y_train,X_test,Y_test,text):
    text.delete(1.0,tk.END)
    text.insert(tk.END, "\n\nIMPORTING AdaBoost"+"\nProcessing this might take a while...","bold")
    text.update_idletasks()
    from sklearn.ensemble import AdaBoostClassifier
    ABC=AdaBoostClassifier()
    text.insert(tk.END, "\n\n Number of Features for Training : "+str(len(X_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Labels for Training : "+str(len(Y_train)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n *** Training AdaBoost using the above Features and Labels ***","bold")
    text.update_idletasks()
    ABC.fit(X_train, Y_train)
    text.insert(tk.END, "\n\n Number of Test Features : "+str(len(X_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Predicting the Test Labels for the above Test Features using AdaBoost ","bold")
    text.update_idletasks()
    Y_pred=ABC.predict(X_test)
    text.insert(tk.END, "\n\n Number of Actual Labels : "+str(len(Y_test)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of Test Labels Predicted by AdaBoost --> "+str(len(Y_pred)),"bold")
    text.insert(tk.END, "\n\n ---------------------------------------------------")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Number of LABELS MATCHED : "+str(accuracy_score(Y_test,Y_pred,normalize=False)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Calculating Accuracy of AdaBoost = Label Matched/Actual Labels ","bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n Accuracy Score : "+str(accuracy_score(Y_test,Y_pred,normalize=True)),"bold")
    text.update_idletasks()
    text.insert(tk.END, "\n\n AdaBoost report \n"+classification_report(Y_test,Y_pred),"bold")
    text.update_idletasks()
    roc_curve_acc(Y_test, Y_pred,'ABC')
    if typ=="s":
       plt.show()
    elif typ=="a":
       pass 


###########################################
top = tk.Tk()
top.title("CREDIT CARD FRAUD DETECTION")
label =tk.Label(master=top,font=("Helvetica", 16), text="CREDIT CARD FRAUD DETECTION")
label.pack()
text = tk.Text(master=top,font="Helvetica 12")
text.pack(side=tk.RIGHT)
bop = tk.Frame()
bop.pack(side=tk.LEFT)
viewdataset = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'blue', foreground = '#eeeeff',text="View Dataset", command=cbc("1",text)).pack()
describeDataset = tk.Button(bop,font=("Helvetica",10,"bold"), background = 'blue', foreground = '#eeeeff',text="Describe Dataset", command=cbc("2",text)).pack()
amount_transaction_hist = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'blue', foreground = '#eeeeff',text="Transactions VS Amount in Histogram", command=cbc("3",text)).pack()
#amount_transaction_textt = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'blue', foreground = '#eeeeff',text="Transactions VS Amount in Normal text", command=cbc("3_1",text)).pack() 
#time_transaction_hist = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'blue', foreground = '#eeeeff',text="Transactions VS TIme in Histogram", command=cbc("4",text)).pack()
#time_transaction_textt = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'blue', foreground = '#eeeeff',text="Transactions VS Time in Normal text", command=cbc("4_1",text)).pack() 
#time_and_Amount = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'blue', foreground = '#eeeeff',text="Amount VS Time Histogram", command=cbc("5",text)).pack() 
##################ALGORITHMS#####################
text.insert(tk.END, "\n Please wait, importing required libraries.\n","bold")
text.update_idletasks()
text.insert(tk.END,"\n IMPORTING PANDAS" ,"bold")
text.update_idletasks()
import pandas as pd 
text.insert(tk.END,"\n IMPORTING train_test_split from sklearn.model_selection" ,"bold")
from sklearn.model_selection import train_test_split
text.insert(tk.END,"\n IMPORTING MATPLOTLIB PYPLOT" ,"bold")
text.update_idletasks()
import matplotlib.pyplot as plt
text.insert(tk.END,"\n IMPORTING SKLEARN METRICS" ,"bold")
text.update_idletasks()
from sklearn.metrics import accuracy_score,classification_report, roc_auc_score, roc_curve, auc
from itertools import cycle
text.insert(tk.END,"\n IMPORTING CONFUSION MATRIX" ,"bold")
text.update_idletasks()
from sklearn.metrics import confusion_matrix
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange','black','pink'])
text.insert(tk.END,"\n LOADING THE DATASET" ,"bold")
text.insert(tk.END,"\n-----------------------\n" ,"bold")
text.update_idletasks()
df = pd.read_csv("../creditcardfraud/creditcard.csv")
text.insert(tk.END,"\n\n DATASET LOADED SUCCESSFULY","bold")
text.insert(tk.END,"\n--------------------------------\n" ,"bold")
text.update_idletasks()


X=df[['V1','V2','V3','V4','V5','V6','V7','V9','V10','V11','V12','V14','V16','V17','V18','V19','V21']]
Y=df["Class"]
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.10, random_state=0)
    
gaussian = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'green', foreground = '#eeeeff',text="Gaussian Naive Bayes(Gaussian NB) Classfier", command=func("6",X_train,Y_train,X_test,Y_test,text)).pack() 
randomforrest = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'green', foreground = '#eeeeff',text="Random Forest Classifier", command=func("7",X_train,Y_train,X_test,Y_test,text)).pack() 
logicalregression = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'green', foreground = '#eeeeff',text="Logistic Regression Classfier", command=func("8",X_train,Y_train,X_test,Y_test,text)).pack() 
Svc = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'green', foreground = '#eeeeff',text="Support Vector Classfier", command=func("9",X_train,Y_train,X_test,Y_test,text)).pack() 
kNeighbours = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'green', foreground = '#eeeeff',text="KNeighbours Classfier", command=func("10",X_train,Y_train,X_test,Y_test,text)).pack() 
Decisiontree = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'green', foreground = '#eeeeff',text="Decision Tree Classfier", command=func("11",X_train,Y_train,X_test,Y_test,text)).pack() 
Extratree = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'green', foreground = '#eeeeff',text="Extra Tree Classfier", command=func("12",X_train,Y_train,X_test,Y_test,text)).pack() 
Adaboost = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'green', foreground = '#eeeeff',text="AdaBoost Classfier", command=func("13",X_train,Y_train,X_test,Y_test,text)).pack() 
Compare_All_Algorithms = tk.Button(bop,font=("Helvetica",10,"bold"),background = 'green', foreground = '#eeeeff',text="Analyse all algorithms", command=func("14",X_train,Y_train,X_test,Y_test,text)).pack() 

tk.Button(bop, text='Exit',font=("Helvetica",10,"bold"),background = 'red', foreground = '#eeeeff', command=top.destroy).pack()
top.mainloop()

