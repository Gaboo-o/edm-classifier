"""Evaluate pooled or attention single-label parent-genre classifiers."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from edm_classifier.training.train import (
    DEFAULT_DATA_DIR, DEFAULT_RUNS_DIR, ENCODER_MODEL_NAMES, EmbeddingDataset,
    EmbeddingResolver, build_model, choose_device, collate_sequences, load_classes, load_jsonl,
)

def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text("",encoding="utf-8"); return
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

@torch.no_grad()
def predict(model, loader, device, representation):
    model.eval(); probs=[]; pred=[]; true=[]
    for batch in loader:
        if representation == "sequence":
            x, mask, y = batch; logits = model(x.to(device), mask.to(device))
        else:
            x, y = batch; logits = model(x.to(device))
        p=torch.softmax(logits,dim=1); probs.append(p.cpu().numpy()); pred.append(torch.argmax(p,dim=1).cpu().numpy()); true.append(y.numpy())
    return np.concatenate(probs),np.concatenate(pred),np.concatenate(true)

def class_metrics(true,pred,classes):
    n=len(classes); matrix=np.zeros((n,n),dtype=np.int64)
    for t,p in zip(true,pred): matrix[int(t),int(p)]+=1
    rows=[]
    for i,item in enumerate(classes):
        tp=int(matrix[i,i]); fp=int(matrix[:,i].sum()-tp); fn=int(matrix[i,:].sum()-tp); support=int(matrix[i,:].sum()); predicted=int(matrix[:,i].sum())
        precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0; f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
        rows.append({"index":i,"id":item["id"],"label":item.get("label",item["id"]),"source":item.get("source"),"support":support,"predicted":predicted,"precision":precision,"recall":recall,"f1":f1,"correct":tp})
    return rows,matrix

def aggregate_metrics(rows,matrix):
    support=np.asarray([r["support"] for r in rows],dtype=np.float64); f1=np.asarray([r["f1"] for r in rows],dtype=np.float64); supported=support>0
    retained=np.asarray([r["id"]!="other" for r in rows],dtype=bool)&supported; total=int(matrix.sum()); correct=int(np.trace(matrix))
    return {"accuracy":correct/total if total else 0.0,"macro_f1":float(f1[supported].mean()) if supported.any() else 0.0,"macro_f1_excluding_other":float(f1[retained].mean()) if retained.any() else 0.0,"weighted_f1":float(np.sum(f1*support)/support.sum()) if support.sum() else 0.0,"samples":total,"correct":correct,"supported_classes":int(supported.sum()),"retained_supported_classes":int(retained.sum())}

def confusion_rows(matrix,classes,normalized):
    out=[]
    for i,item in enumerate(classes):
        total=int(matrix[i].sum()); row={"true_id":item["id"],"true_label":item.get("label",item["id"])}
        for j,other in enumerate(classes): row[other["id"]]=float(matrix[i,j]/total) if normalized and total else (0.0 if normalized else int(matrix[i,j]))
        out.append(row)
    return out

def top_confusions(matrix,classes,top_k):
    out=[]
    for i,item in enumerate(classes):
        support=int(matrix[i].sum()); correct=int(matrix[i,i]); ranked=sorted(((j,int(matrix[i,j])) for j in range(len(classes)) if j!=i),key=lambda x:x[1],reverse=True)
        for rank,(j,count) in enumerate(ranked[:top_k],1):
            other=classes[j]; out.append({"true_id":item["id"],"true_label":item.get("label",item["id"]),"support":support,"correct":correct,"accuracy":correct/support if support else 0.0,"rank":rank,"confused_with_id":other["id"],"confused_with_label":other.get("label",other["id"]),"count":count,"fraction_of_true_class":count/support if support else 0.0})
    return out

def plot_confusion(path,matrix,classes):
    norm=matrix.astype(np.float64); row_sum=norm.sum(axis=1,keepdims=True); norm=np.divide(norm,row_sum,out=np.zeros_like(norm),where=row_sum>0); labels=[x.get("label",x["id"]) for x in classes]
    fig,ax=plt.subplots(figsize=(13,11)); image=ax.imshow(norm,aspect="auto",vmin=0.0,vmax=1.0); fig.colorbar(image,ax=ax,label="Fraction of true class")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=90); ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.set_xlabel("Predicted class"); ax.set_ylabel("True class"); ax.set_title("Single-label parent genre confusion matrix"); fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)

def parse_args():
    p=argparse.ArgumentParser(description="Evaluate single-label parent classifier."); p.add_argument("--split",choices=["regular","artist"],required=True); p.add_argument("--model",choices=["linear","mlp","attention"],default="mlp"); p.add_argument("--encoder",choices=sorted(ENCODER_MODEL_NAMES),default="discogs"); p.add_argument("--embeddings-dir",type=Path,default=None); p.add_argument("--data-dir",type=Path,default=DEFAULT_DATA_DIR); p.add_argument("--runs-dir",type=Path,default=DEFAULT_RUNS_DIR); p.add_argument("--batch-size",type=int,default=None); p.add_argument("--device",default="auto"); p.add_argument("--num-workers",type=int,default=0); p.add_argument("--top-k",type=int,default=3); return p.parse_args()

def main():
    args=parse_args(); classes=load_classes(args.data_dir/"classes.json"); class_count=len(classes); run_dir=args.runs_dir/f"{args.encoder}_{args.split}_{args.model}"; checkpoint_path=run_dir/"model.pt"; norm_path=run_dir/"normalization.npz"
    if not checkpoint_path.is_file(): raise SystemExit(f"missing checkpoint: {checkpoint_path}")
    if not norm_path.is_file(): raise SystemExit(f"missing normalization: {norm_path}")
    device=choose_device(args.device); checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=False)
    if checkpoint.get("encoder","discogs") != args.encoder: raise ValueError(f"checkpoint encoder is {checkpoint.get('encoder')!r}, but --encoder is {args.encoder!r}")
    expected=[x["id"] for x in classes]
    if checkpoint.get("class_ids") != expected: raise ValueError("checkpoint classes do not match current classes.json")
    representation=checkpoint.get("representation","sequence" if checkpoint["model_name"]=="attention" else "pooled")
    norm=np.load(norm_path,allow_pickle=False); mean=np.asarray(norm["mean"],dtype=np.float32); std=np.asarray(norm["std"],dtype=np.float32)
    model=build_model(checkpoint["model_name"],input_dim=int(checkpoint["input_dim"]),output_dim=int(checkpoint["output_dim"]),hidden_dim=int(checkpoint.get("hidden_dim",512)),dropout=float(checkpoint.get("dropout",0.2))).to(device); model.load_state_dict(checkpoint["state_dict"])
    checkpoint_source=checkpoint.get("embedding_source"); checkpoint_dir=None
    if isinstance(checkpoint_source,dict):
        value=checkpoint_source.get("directory") or checkpoint_source.get("pooled_dir")
        if isinstance(value,str) and value: checkpoint_dir=Path(value)
    resolver=EmbeddingResolver(args.encoder,args.embeddings_dir or checkpoint_dir,representation=representation)
    rows=load_jsonl(args.data_dir/args.split/"test.jsonl"); ds=EmbeddingDataset(rows,mean=mean,std=std,class_count=class_count,resolver=resolver); batch_size=args.batch_size or (16 if representation=="sequence" else 512); collate=collate_sequences if representation=="sequence" else None
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=args.num_workers,collate_fn=collate,pin_memory=torch.cuda.is_available())
    probs,pred,true=predict(model,loader,device,representation); per_class,matrix=class_metrics(true,pred,classes); aggregate=aggregate_metrics(per_class,matrix)
    write_csv(run_dir/"test_per_class.csv",per_class); write_csv(run_dir/"confusion_counts.csv",confusion_rows(matrix,classes,False)); write_csv(run_dir/"confusion_row_normalized.csv",confusion_rows(matrix,classes,True)); write_csv(run_dir/"top_confusions.csv",top_confusions(matrix,classes,args.top_k)); plot_confusion(run_dir/"confusion_heatmap.png",matrix,classes)
    max_prob=probs.max(axis=1); confidence={"mean_max_probability":float(max_prob.mean()),"median_max_probability":float(np.median(max_prob)),"correct_mean_max_probability":float(max_prob[pred==true].mean()) if np.any(pred==true) else 0.0,"incorrect_mean_max_probability":float(max_prob[pred!=true].mean()) if np.any(pred!=true) else 0.0}
    report={"run_name":f"{args.encoder}_{args.split}_{args.model}","task":"single_label_parent_genre","encoder":checkpoint.get("encoder"),"encoder_model":checkpoint.get("encoder_model"),"embedding_source":resolver.describe(),"representation":representation,"pooling":checkpoint.get("pooling"),"split":args.split,"model":args.model,"checkpoint_epoch":checkpoint.get("epoch"),"class_weight_mode":checkpoint.get("class_weight_mode"),"class_count":class_count,"test":aggregate,"confidence":confidence,"outputs":{"per_class":str(run_dir/"test_per_class.csv"),"confusion_counts":str(run_dir/"confusion_counts.csv"),"confusion_normalized":str(run_dir/"confusion_row_normalized.csv"),"top_confusions":str(run_dir/"top_confusions.csv"),"heatmap":str(run_dir/"confusion_heatmap.png")}}
    report_path=run_dir/"evaluation_report.json"; report_path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"Single-label evaluation\n  split:                  {args.split}\n  model:                  {args.model}\n  encoder:                {args.encoder}\n  representation:         {representation}\n  embeddings:             {resolver.root}\n  samples:                {aggregate['samples']}\n  accuracy:               {aggregate['accuracy']:.4f}\n  macro F1:               {aggregate['macro_f1']:.4f}\n  macro F1 excluding O:   {aggregate['macro_f1_excluding_other']:.4f}\n  weighted F1:            {aggregate['weighted_f1']:.4f}\n\nReport: {report_path}")
if __name__=="__main__": main()
