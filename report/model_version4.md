## Model architecture
![ex_screenshot](../assets/model_version4.JPG)



- our model( bidirectional + concat-pooling )

|   |string equal|dfa equal|membership equal|total|
|------|---|---|---|---|
|star0|680|0|76|37.8%|
|star1|186|54|167|20.37%|
|star2|82|30|153|13.26%|
|star3|27|27|86|7%|


- our model( bidirectional + concat-pooling +attention(only positive samples))

|   |string equal|dfa equal|membership equal|total|
|------|---|---|---|---|
|star0|1053|7|163|61.15%|
|star1|384|103|303|39.5%|
|star2|218|76|305|29.97%|
|star3|103|61|222|19.41%|



