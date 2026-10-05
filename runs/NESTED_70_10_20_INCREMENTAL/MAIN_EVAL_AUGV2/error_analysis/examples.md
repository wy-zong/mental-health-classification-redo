# v2（MAIN_EVAL_AUGV2）質性範例與錯誤分析

> ⚠️ 文本來自心理健康貼文資料集，屬敏感內容。此處已截短至 300 字元，放進論文前必須再改寫或摘要，不可逐字引用。

產生方式：`src/18_custom_prompt_error_analysis.py`，固定 seed = 42 抽樣。完整樣本見同目錄的 jsonl。

條件：C1　無檢索・基礎 prompt（`norag_base`）；C2　原文檢索・基礎 prompt（`rag_noaug_base`）；C3　原文＋擴寫檢索・基礎 prompt（`rag_augv2_base`）；C4　原文檢索・最佳化 prompt（`rag_noaug_optimized`）；C5　原文＋擴寫檢索・最佳化 prompt（`rag_augv2_optimized`）。RAG 條件 top-k = 5；下表「檢索標籤」指 top-1。

## RAG 的效果（C1　無檢索・基礎 prompt → C2　原文檢索・基礎 prompt）

| 群組 | n | C1 為 INVALID | C2 為 INVALID | C2 預測＝檢索標籤 | 檢索標籤正確 |
|---|---|---|---|---|---|
| RAG 有幫助（C1 錯 → C2 對） | 439 | 290 | 0 | 407 | 407 |
| RAG 有害（C1 對 → C2 錯） | 215 | 0 | 133 | 48 | 123 |

## 檢索標籤與預測的關係（C2–C5）

| 條件 | 檢索標籤正確率 | 照抄率（全部） | 照抄率（有效回覆） | 檢索對時準確率 | 檢索錯時準確率 | 檢索錯時照抄錯標籤 |
|---|---|---|---|---|---|---|
| C2　原文檢索・基礎 prompt | 0.7873 | 0.6937 | 0.8441 | 0.7616 | 0.2471 | 0.4424 |
| C3　原文＋擴寫檢索・基礎 prompt | 0.7738 | 0.7322 | 0.8256 | 0.8040 | 0.2920 | 0.4867 |
| C4　原文檢索・最佳化 prompt | 0.7873 | 0.8433 | 0.8610 | 0.8989 | 0.2518 | 0.6376 |
| C5　原文＋擴寫檢索・最佳化 prompt | 0.7738 | 0.8559 | 0.8698 | 0.9075 | 0.2279 | 0.6792 |

## 範例：RAG 有幫助（439 筆中抽 5 筆）

#### `test_01235`（true = Depression）

- 輸入：I am 22. My whole life I have struggled with mental illness. Everyday I struggle with severe anxiety and depression. Sometimes I also struggle with suicide. Its gotten so bad that I finally convinced my parents to be okay with me going to the doctors to try medication. My appointment is coming up an …
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **INVALID**（錯誤），raw response：`**Depression** The text describes a person struggling with severe anxiety and de …`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **Depression**（正確），raw response：`Based on the text, I would classify it as: **Depression** The`
  - 檢索 top-1：label = Depression，similarity = 0.7566，id = `val_confirm_00086`
  - 檢索 top-5 標籤：Depression（0.7566）、Depression（0.7285）、Depression（0.7217）、Anxiety（0.6923）、Anxiety（0.6910）
  - 檢索文本（top-1）：I cannot tell if my parents love me. Yeah they say it all the time but they do not really show it. they have made it clear that I am a bad person in their eyes. My dad has said before that he regrets having me and I suspect my mom agrees. I do not blame them. I am such a shitty son to them. How coul …

#### `test_02357`（true = Anxiety）

- 輸入：Toenail Melanoma? This sounds so stupid, but I’m worried about toenail melanoma. I recently removed my nail polish and noticed that on one of my toenails, a little less than half of it is a light brown color. The pigment starts from the edge of the nail so it’s not a sole brown line. It just looks l …
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **Normal**（錯誤），raw response：`I would classify this text as: **Normal** The text appears to be a`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **Anxiety**（正確），raw response：`Based on the text, I would classify it as: **Anxiety** The`
  - 檢索 top-1：label = Anxiety，similarity = 0.7623，id = `train_04619`
  - 檢索 top-5 標籤：Anxiety（0.7623）、Anxiety（0.5886）、Anxiety（0.5562）、Anxiety（0.5432）、Anxiety（0.5413）
  - 檢索文本（top-1）：Light brown line on nails I’m 17 years old I had these lines on my left index and middle finger for about two years. I’m feel healthy enough. But I’m freaking out reading about melanoma. true_label is Anxiety

#### `test_02294`（true = Anxiety）

- 輸入：How do you guys manage fever? Please let me know urgent. Thanks!
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **Normal**（錯誤），raw response：`Normal`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **Anxiety**（正確），raw response：`Based on the text, I would classify it as: **Anxiety** The`
  - 檢索 top-1：label = Anxiety，similarity = 0.5446，id = `train_03268`
  - 檢索 top-5 標籤：Anxiety（0.5446）、Anxiety（0.5147）、Anxiety（0.4787）、Anxiety（0.4625）、Normal（0.4474）
  - 檢索文本（top-1）：2 days Fever after Heat Exhaustion Hi guys,i am feeling worried about my child. We went to a farm the other day and it was very hot. after leaving the farm, my child was not feeling well. She got a mild fever when we arrived in our home. 1 day after, she still has a fever and one time it went as hig …

#### `test_00187`（true = Depression）

- 輸入：I have always had suicidal thoughts since I was a teenager but in recent years things have been bad. The last year has been difficult for me as I was made redundant last July. I live with my mother, I am still a virgin at the age of 31, never had a girlfriend or kiss a girl. I suffer from low confid …
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **INVALID**（錯誤），raw response：`I cannot provide a mental health diagnosis. If you are experiencing suicidal tho …`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **Depression**（正確），raw response：`Based on the text, I would classify it as: **Depression** The`
  - 檢索 top-1：label = Bipolar，similarity = 0.7258，id = `test_01694`
  - 檢索 top-5 標籤：Bipolar（0.7258）、Depression（0.7080）、Depression（0.7076）、Depression（0.7075）、Depression（0.7034）
  - 檢索文本（top-1）：There is no help for the functionally depressed / suicidal I am depressed, I’ve been depressed for years. Right now I am in the midst of an especially bad depressive episode and I honestly I won’t fight any feelings if I want to kill myself and have a pretty solid plan when it happens. But I keep go …

#### `test_02017`（true = Bipolar）

- 輸入：Good news! I just got home from an appointment with my psychiatrist and she’s told me that I’ve been improving 😍 She reduced my 3 meds to just Quetiapine and I’m really happy today 🧡
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **Normal**（錯誤），raw response：`I would classify this text as: **Normal** The text indicates a positive emotiona …`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **Bipolar**（正確），raw response：`Based on the text, I would classify it as: **Bipolar**`
  - 檢索 top-1：label = Bipolar，similarity = 0.5997，id = `train_05706`
  - 檢索 top-5 標籤：Bipolar（0.5997）、Bipolar（0.5974）、Bipolar（0.5936）、Depression（0.5810）、Bipolar（0.5711）
  - 檢索文本（top-1）：Finally found the right medication cocktail ?? I'm so happy and I just wanted to share. After years of misdiagnoses and bad meds, I finally found the right ones! I feel like myself again! It has effected my sex drive but I'm working on that. But all and all I feel stable and happy! I wish the best f …

## 範例：RAG 有害（215 筆中抽 5 筆）

#### `test_02088`（true = Depression）

- 輸入：I do not want to offend anyone, so please know I am just asking for advice/wisdom/experience, not trying to be hurtful. Do you know when you are not yourself? My husband thinks everything is fine, but he is so grumpy with all of us (3 kids), short tempered and not fun to be around (usually is amazin …
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **Depression**（正確），raw response：`**Depression** The text describes a situation where the writer is experiencing f …`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **Bipolar**（錯誤），raw response：`Based on the text, I would classify it as: **Bipolar**`
  - 檢索 top-1：label = Bipolar，similarity = 0.7583，id = `train_00628`
  - 檢索 top-5 標籤：Bipolar（0.7583）、Bipolar（0.7512）、Bipolar（0.7441）、Bipolar（0.7438）、Depression（0.7432）
  - 檢索文本（top-1）：Loved Ones with Bipolar I am unsure if there already is a post on how to cope with loved ones with bipolar. Please let me know if there is. My husband was hospitalized about 5 months ago and it's been a roller coaster since with finding which meds work, finding a permanent psychiatrist, going to the …

#### `test_00455`（true = Normal）

- 輸入：homework
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **Normal**（正確），raw response：`Normal`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **INVALID**（錯誤），raw response：`The text "homework" does not contain any emotional or mental health-related cont …`
  - 檢索 top-1：label = Normal，similarity = 0.4630，id = `test_02342`
  - 檢索 top-5 標籤：Normal（0.4630）、Depression（0.3310）、Normal（0.3249）、Normal（0.3055）、Normal（0.3010）
  - 檢索文本（top-1）：got ta stop turning in homework late would be gettin an a in information system if it wasnt for the few late assigments true_label is Normal

#### `test_01197`（true = Anxiety）

- 輸入：I have finished all my tasks from yesterday, but why is this heart restless? feel like something bad is coming?! ......
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **Anxiety**（正確），raw response：`I cannot provide a mental health diagnosis. If you are experiencing anxiety or o …`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **INVALID**（錯誤），raw response：`Based on the reference content and the provided text, I would classify the text  …`
  - 檢索 top-1：label = Anxiety，similarity = 0.7274，id = `train_01168`
  - 檢索 top-5 標籤：Anxiety（0.7274）、Anxiety（0.6642）、Anxiety（0.6558）、Anxiety（0.6317）、Anxiety（0.6242）
  - 檢索文本（top-1）：Oh my God, I don't know why my heart is restless :( true_label is Anxiety

#### `test_02180`（true = Normal）

- 輸入：Btw, if you are pregnant, can you have sex or not?
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **Normal**（正確），raw response：`Normal`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **INVALID**（錯誤），raw response：`Based on the content of the text, I would classify it as: **An`
  - 檢索 top-1：label = Anxiety，similarity = 0.4234，id = `test_01453`
  - 檢索 top-5 標籤：Anxiety（0.4234）、Anxiety（0.3822）、Anxiety（0.3797）、Normal（0.3726）、Bipolar（0.3429）
  - 檢索文本（top-1）：Can a girl get pregnant from a towel? so after I masturbated I cleaned myself with a towel, forgot to take the towel from the bathroom and my sister went in and used the towel to dry herself. Is it possible she could be pregnant? I'm freaking out true_label is Anxiety

#### `test_00210`（true = Bipolar）

- 輸入：I never knew it was manic episodes I never knew going off texting someone a million texts was me being manic Telling them borderline cruel things was being manic. I didn't know it could all escalate. That going through more trauma could affect me the way it has. Dammit I thought I had things under c …
- **C1**（無檢索・基礎 prompt，`norag_base`）→ 預測 **Bipolar**（正確），raw response：`**Bipolar** The text describes a pattern of intense emotional episodes, includin …`
- **C2**（原文檢索・基礎 prompt，`rag_noaug_base`）→ 預測 **INVALID**（錯誤），raw response：`Based on the content of the text, I would classify it as: **B`
  - 檢索 top-1：label = Bipolar，similarity = 0.8030，id = `train_01827`
  - 檢索 top-5 標籤：Bipolar（0.8030）、Bipolar（0.7967）、Bipolar（0.7966）、Bipolar（0.7931）、Bipolar（0.7920）
  - 檢索文本（top-1）：Got diagnosed and am now spiraling So I found out I was bipolar by looking it up because it popped in my head one day and ADHD didn't explain everything. So I did that, realized it matched perfectly, talked to my psych, and it turned out he already thought I had a mood disorder but didn't know which …

## C4　原文檢索・最佳化 prompt（`rag_noaug_optimized`）的主要錯誤類型

| 真實 | 預測 | 筆數 | 佔錯誤 | 佔該真實類別 |
|---|---|---|---|---|
| Normal | Anxiety | 90 | 0.189 | 0.180 |
| Depression | Bipolar | 70 | 0.147 | 0.140 |
| Depression | Anxiety | 63 | 0.132 | 0.126 |
| Normal | Bipolar | 47 | 0.099 | 0.094 |
| Normal | Depression | 46 | 0.096 | 0.092 |
| Bipolar | Depression | 35 | 0.073 | 0.070 |
| Bipolar | Anxiety | 32 | 0.067 | 0.064 |
| Anxiety | Bipolar | 22 | 0.046 | 0.044 |

### C4：Normal → Anxiety（90 筆中抽 3 筆）

#### `test_02127`（true = Normal）

- 輸入：overwhelmed and enjoy it
- **C4**（原文檢索・最佳化 prompt，`rag_noaug_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Depression，similarity = 0.4713，id = `train_02085`
  - 檢索 top-5 標籤：Depression（0.4713）、Anxiety（0.4557）、Anxiety（0.4263）、Bipolar（0.4235）、Anxiety（0.4195）
  - 檢索文本（top-1）：I cannot say what I am feeling is a *good* feeling, but its approaching, and I am a little overwhelmed. I part of me just wants to sit here and feel it and not take it for granted and the other part wants to cry and shut it down. Like, I do not know what to do. Either cry because I am overwhelmed or …

#### `test_01654`（true = Normal）

- 輸入：I have to first start off by saying my voice is not high-pitched. My inflection occasionally goes up, especially if I'm excited about something. I'm sort of expressive with my hands while talking, though I don't think more than anyone else, male or female. She tells me this is the only thing that is …
- **C4**（原文檢索・最佳化 prompt，`rag_noaug_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Depression，similarity = 0.4188，id = `train_05358`
  - 檢索 top-5 標籤：Depression（0.4188）、Anxiety（0.4111）、Anxiety（0.3882）、Normal（0.3870）、Normal（0.3760）
  - 檢索文本（top-1）：my voice is soo deep how to improve In childhood i had a great voice but after puberty it became very deep. Now iam 20 yo male. The voice is so deep that i have to put lot of effort to be heard in a place which has any background noise. My friends also point out that its too deep. I wish my voice ha …

#### `test_00627`（true = Normal）

- 輸入：feel like she slept the day away not looking forward to any more bout with my gallbladder at least i have pill now for the pain
- **C4**（原文檢索・最佳化 prompt，`rag_noaug_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Anxiety，similarity = 0.5127，id = `train_00585`
  - 檢索 top-5 標籤：Anxiety（0.5127）、Anxiety（0.4717）、Anxiety（0.4449）、Normal（0.4422）、Anxiety（0.4375）
  - 檢索文本（top-1）：Crippling medical anxiety So I haven’t been to see a doctor in about two years because I was insanely nervous about what they might find. I was hit by a car Sunday so I finally went to see a doctor today. While talking to the doctor I mentioned I get sharp pain right under my ribs often. She said it …

### C4：Depression → Bipolar（70 筆中抽 3 筆）

#### `test_02071`（true = Depression）

- 輸入：i ve been on this for week and still barely have an appetite i even got nauseous going to the grocery store today is this normal ha anyone else experienced this it s been great for my mood and communication but it mess with my diet i don t even want to consume food really keep in mind though i did u …
- **C4**（原文檢索・最佳化 prompt，`rag_noaug_optimized`）→ 預測 **Bipolar**（錯誤），raw response：`Bipolar`
  - 檢索 top-1：label = Bipolar，similarity = 0.6553，id = `train_01163`
  - 檢索 top-5 標籤：Bipolar（0.6553）、Bipolar（0.6125）、Bipolar（0.6058）、Depression（0.6038）、Depression（0.5968）
  - 檢索文本（top-1）：Depakote &amp; Food So I (23F) recently started Depakote (as of today it's been about a week of taking it) and it's working really well for my mood swings and general bipolar I issues. One thing I've noticed is I don't really want to eat. I just don't get hungry? I've heard everyone talk about incre …

#### `test_00351`（true = Depression）

- 輸入：hey guy i wanted to throw this out there and see if any of you would be interested i m looking to start a group zoom meeting for people with anxiety depression bipolar etc it s going to be totally free we can share our story meet up once a week and just talk about how we are doing our feeling really …
- **C4**（原文檢索・最佳化 prompt，`rag_noaug_optimized`）→ 預測 **Bipolar**（錯誤），raw response：`Bipolar`
  - 檢索 top-1：label = Bipolar，similarity = 0.6711，id = `train_03158`
  - 檢索 top-5 標籤：Bipolar（0.6711）、Bipolar（0.6453）、Depression（0.6452）、Bipolar（0.6092）、Bipolar（0.6055）
  - 檢索文本（top-1）：Any discord’s for depression/bipolar sufferers? Just wondering if there’s a place where you can talk in real time (chat room) with people who are going through the same things someone like me is going through. (Can’t got outta bed, haven’t showered p, general feelings of worthlessness. true_label is …

#### `test_00540`（true = Depression）

- 輸入：I am depressed but I do not want to kill myself. I am just depressed. I do not complain about it. Sometimes I feel good. Sometimes I feel bad. I get really bad manic depression sometimes but my face ALWAYS gets commented on. "Cheer up" "you look lile you are about to cry" even though I am not... I c …
- **C4**（原文檢索・最佳化 prompt，`rag_noaug_optimized`）→ 預測 **Bipolar**（錯誤），raw response：`Bipolar`
  - 檢索 top-1：label = Depression，similarity = 0.6822，id = `train_03588`
  - 檢索 top-5 標籤：Depression（0.6822）、Depression（0.6773）、Bipolar（0.6588）、Depression（0.6533）、Depression（0.6532）
  - 檢索文本（top-1）：Hi, i want to apologize for my English in advance, I am not a native speaker, just a 21 years old guy, that needs to ventilate somewhere.I have problems with my mental health, I am sad all the time, alone, depressed and every single day, when i wake up i struggle to get out of my bed. But i somehow  …

### C4：Depression → Anxiety（63 筆中抽 3 筆）

#### `test_00626`（true = Depression）

- 輸入：today ha just been so shitty it s so busy at the store i work at and i just constantly feel like i can t breath today i m also so paranoid because i ve been texting my family literally all day and nobody s gotten back to me so i m stupidly paranoid about something bad happening to them
- **C4**（原文檢索・最佳化 prompt，`rag_noaug_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Depression，similarity = 0.6450，id = `train_01153`
  - 檢索 top-5 標籤：Depression（0.6450）、Anxiety（0.6032）、Anxiety（0.6021）、Anxiety（0.5972）、Anxiety（0.5869）
  - 檢索文本（top-1）：i don t know how i can feel this horrible and unable to breathe so badly and this only be anxiety i genuinely feel like i m going to pas out and i have nothing to be anxious about is this really what anxiety feel like i can t take a deep breath this is so awful true_label is Depression

#### `test_01968`（true = Depression）

- 輸入：I am trying to do strength training but its almost impossible to get myself to actually do more than a couple chin ups or something and I do not know how to get myself to do more How do get myself to workout when I have no energy?
- **C4**（原文檢索・最佳化 prompt，`rag_noaug_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Based on the text, I would classify it as: **Anxiety** The`
  - 檢索 top-1：label = Depression，similarity = 0.4147，id = `val_search_00051`
  - 檢索 top-5 標籤：Depression（0.4147）、Anxiety（0.4121）、Bipolar（0.4074）、Bipolar（0.3995）、Depression（0.3920）
  - 檢索文本（top-1）：I can barely force myself to get out of bed let alone get up early, eat healthty, go for a run/workout and then go to hell(work) i work 14pm-22pm and 6am-14pm every other week so the week i have to get up early I am just too tired to do anything after work and the week i work late i do not want to g …

#### `test_02247`（true = Depression）

- 輸入：i f 0 lb think i m having heart burn right now though i m not sure at around 00 today i suddenly started getting a weird chest pain it s not severe pain more like a mild dull stabbing pain that only last in certain position if i lay a certain way the chest pain will go away however i feel the pain a …
- **C4**（原文檢索・最佳化 prompt，`rag_noaug_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Anxiety，similarity = 0.7368，id = `train_04483`
  - 檢索 top-5 標籤：Anxiety（0.7368）、Anxiety（0.7142）、Anxiety（0.7131）、Anxiety（0.6982）、Anxiety（0.6822）
  - 檢索文本（top-1）：Advice? Life's become quite hard worrying about heart attacks I'm 19, 6'1 and about 75kg. I've drank loads of energy drinks in my life and I'm a bit lethargic, I don't do anything deliberately to exercise. But I'm not fat and I don't eat too badly and whatever. No family history of heart attacks at  …

## C5　原文＋擴寫檢索・最佳化 prompt（`rag_augv2_optimized`）的主要錯誤類型

| 真實 | 預測 | 筆數 | 佔錯誤 | 佔該真實類別 |
|---|---|---|---|---|
| Normal | Anxiety | 101 | 0.205 | 0.202 |
| Depression | Anxiety | 74 | 0.150 | 0.148 |
| Depression | Bipolar | 66 | 0.134 | 0.132 |
| Normal | Depression | 44 | 0.089 | 0.088 |
| Bipolar | Anxiety | 37 | 0.075 | 0.074 |
| Bipolar | Depression | 36 | 0.073 | 0.072 |
| Normal | Bipolar | 34 | 0.069 | 0.068 |
| Anxiety | Depression | 24 | 0.049 | 0.048 |

### C5：Normal → Anxiety（101 筆中抽 3 筆）

#### `test_01613`（true = Normal）

- 輸入：i forgot my phone in my car but i m too scared to go outside and get it
- **C5**（原文＋擴寫檢索・最佳化 prompt，`rag_augv2_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Anxiety，similarity = 0.4610，id = `train_02368`
  - 檢索 top-5 標籤：Anxiety（0.4610）、Anxiety（0.4133）、Anxiety（0.3950）、Anxiety（0.3931）、Normal（0.3907）
  - 檢索文本（top-1）：I hate getting anxiety attacks over small shit. so, just a second ago my phone stopped randomly working. I could open apps but couldn’t scroll or do anything once I was in the app. I got super anxious and was at the verge of TEARS because Im a fucking idiot that was like “oh god now my phones not wo …

#### `test_00343`（true = Normal）

- 輸入：Are you clean yet?
- **C5**（原文＋擴寫檢索・最佳化 prompt，`rag_augv2_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Normal，similarity = 0.4933，id = `aug_train_02500_g00`
  - 檢索 top-5 標籤：Normal（0.4933）、Bipolar（0.4742）、Normal（0.4633）、Normal（0.4588）、Normal（0.4561）
  - 檢索文本（top-1）：Let's tackle the cleaning before we relax, shall we? true_label is Normal

#### `test_00170`（true = Normal）

- 輸入：warlach curse ye have fun at it i miss doing online pr for paramount pic
- **C5**（原文＋擴寫檢索・最佳化 prompt，`rag_augv2_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Normal，similarity = 0.3666，id = `aug_train_01599_m1`
  - 檢索 top-5 標籤：Normal（0.3666）、Normal（0.3551）、Normal（0.3474）、Normal（0.3350）、Normal（0.3263）
  - 檢索文本（top-1）：Seeking a cinematic pick-me-up for tomorrow's celebration. true_label is Normal

### C5：Depression → Anxiety（74 筆中抽 3 筆）

#### `test_01284`（true = Depression）

- 輸入：ha anyone had really intense brain fog this year i mean it s been pretty bad before but i genuinely can t remember stuff and relative time is confusing sometimes a day feel like a week or i ll think i discussed something with someone on monday and it s been week and they ve been waiting for me to fo …
- **C5**（原文＋擴寫檢索・最佳化 prompt，`rag_augv2_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Anxiety，similarity = 0.5903，id = `train_02404`
  - 檢索 top-5 標籤：Anxiety（0.5903）、Depression（0.5855）、Depression（0.5730）、Anxiety（0.5683）、Anxiety（0.5615）
  - 檢索文本（top-1）：What is wrong with me? I’ve always been extremely articulate and smart. I operated on a higher consciousness than most people just in the sense that I was more aware of my surroundings and I thought deeply about things. I was also quick witted and being funny was probably my defining personality tra …

#### `test_01552`（true = Depression）

- 輸入：I have a lot of built in emotions from my childhood up until a break up I just went through. I find myself with a mix of negative emotions and a lot of it is surfacing up to where I am projecting it onto others and sabotaging. I love the happy and positive me. I go to therapy but its not enough to h …
- **C5**（原文＋擴寫檢索・最佳化 prompt，`rag_augv2_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Anxiety，similarity = 0.6343，id = `aug_test_02163_n0`
  - 檢索 top-5 標籤：Anxiety（0.6343）、Bipolar（0.6239）、Depression（0.6158）、Depression（0.6131）、Bipolar（0.6127）
  - 檢索文本（top-1）：Embracing the Emotions Right now, I'm facing a particularly tough time, and it's forced me to confront the fact that I've been sidestepping difficult emotions for as long as I can remember. As I become more aware of this pattern, I'm attempting to let my emotions unfold and process them naturally, b …

#### `test_01901`（true = Depression）

- 輸入：hello all i am sorry i have to make a post for this i am just new to therapy and medication my therapist prescribed me lexapro and told me if that didn t workout or it made me too sick or lightheaded she would switch me to zoloft what is the difference all i can find is horror story on lexapro or pe …
- **C5**（原文＋擴寫檢索・最佳化 prompt，`rag_augv2_optimized`）→ 預測 **Anxiety**（錯誤），raw response：`Anxiety`
  - 檢索 top-1：label = Anxiety，similarity = 0.7280，id = `aug_train_01093_m1`
  - 檢索 top-5 標籤：Anxiety（0.7280）、Anxiety（0.7174）、Anxiety（0.6990）、Anxiety（0.6948）、Anxiety（0.6914）
  - 檢索文本（top-1）：Switching to Zoloft has been a game-changer for me. I was blown away by the contrast with Lexapro - the improvement is nothing short of remarkable. I no longer wake up exhausted and drowsy, nor do I find myself constantly battling weight gain. And, I'm cautiously optimistic about the possibility of  …

### C5：Depression → Bipolar（66 筆中抽 3 筆）

#### `test_00142`（true = Depression）

- 輸入：when you are fucking done with life and going to buy some more liquor in 6 in morning !!normal people be alike tittle
- **C5**（原文＋擴寫檢索・最佳化 prompt，`rag_augv2_optimized`）→ 預測 **Bipolar**（錯誤），raw response：`Bipolar`
  - 檢索 top-1：label = Normal，similarity = 0.4889，id = `val_confirm_01050`
  - 檢索 top-5 標籤：Normal（0.4889）、Bipolar（0.4604）、Bipolar（0.4493）、Normal（0.4447）、Depression（0.4425）
  - 檢索文本（top-1）：wow i just woke up and there's a bag next to my bed. i hate alcohol! wtf happened tonight?!?! thanks everyone who came to my bday dinner true_label is Normal

#### `test_00603`（true = Depression）

- 輸入：The career I chose is in the process of making me very rich, but also extremely depressed and overworked/overjudged to the point that I am having at least one episode a day where I want to curl up and disappear. Its a job that involves social media, fame, and the expectation of perfection. Sometimes …
- **C5**（原文＋擴寫檢索・最佳化 prompt，`rag_augv2_optimized`）→ 預測 **Bipolar**（錯誤），raw response：`Bipolar`
  - 檢索 top-1：label = Bipolar，similarity = 0.7587，id = `train_04473`
  - 檢索 top-5 標籤：Bipolar（0.7587）、Depression（0.7535）、Depression（0.7453）、Depression（0.7432）、Depression（0.7425）
  - 檢索文本（top-1）：I'm depressed Friend of mine, who I valued a lot, just cut all ties with me. Told me that I’m too needy and emotional. They’re not wrong. I’m bipolar and I had a really fucked up childhood which caused a lot of abandonment issues, something I’m in therapy for now. It does make me very emotional at t …

#### `test_01155`（true = Depression）

- 輸入：I work to "save money for college" but i intend on dying before i go to college. I work for basically inherent reason, but i am still pressed to go because everyone else insists that I am going to end up not killing myself. If I was born rich i would at least get to spend my final weeks alive in pea …
- **C5**（原文＋擴寫檢索・最佳化 prompt，`rag_augv2_optimized`）→ 預測 **Bipolar**（錯誤），raw response：`Bipolar`
  - 檢索 top-1：label = Bipolar，similarity = 0.7421，id = `aug_train_03886_m1`
  - 檢索 top-5 標籤：Bipolar（0.7421）、Bipolar（0.7281）、Depression（0.7169）、Depression（0.7040）、Depression（0.7028）
  - 檢索文本（top-1）：I've been grappling with persistent suicidal thoughts and a recent attempt during a severe depressive episode. Instead of dismissing it as a symptom of my condition, I want to walk you through my mindset. 1. The argument from circumstance is convincing. With a bleak job market, a spotty resume, and  …
