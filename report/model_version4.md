# version4 
### concat Pooling 추가
- 모든 time step의 hidden state에 대하여 average와 max를 취해 기존의 final hidden state와 concat하여 사용함  
- concat -> [final_hidden_state; average Pool; max Pool]
