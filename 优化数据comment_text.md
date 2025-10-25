## 移除CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

##  直接判定
List of some words that often appear in toxic comments Sorry about the level of toxicity in it!
toxic_words = ["poop", "crap", "prick", "twat", "wikipedia", "wiki", "hahahahaha", "lol", "bastard", "sluts", "slut", "douchebag", "douche", "blowjob", "nigga", "dumb", "jerk", "wanker", "wank", "penis", "motherfucker", "fucker", "fuk", "fucking", "fucked", "fuck", "bullshit", "shit", "stupid", "bitches", "bitch", "suck", "cunt", "dick", "cocks", "cock", "die", "kill", "gay", "jewish", "jews", "jew", "niggers", "nigger", "faggot", "fag", "asshole"]



## 替换为后者
astericks_words = [('mother****ers', 'motherfuckers'), ('motherf*cking', 'motherfucking'), ('mother****er', 'motherfucker'), ('motherf*cker', 'motherfucker'), ('bullsh*t', 'bullshit'), ('f**cking', 'fucking'), ('f*ucking', 'fucking'), ('fu*cking', 'fucking'), ('****ing', 'fucking'), ('a**hole', 'asshole'), ('assh*le', 'asshole'), ('f******', 'fucking'), ('f*****g', 'fucking'), ('f***ing', 'fucking'), ('f**king', 'fucking'), ('f*cking', 'fucking'), ('fu**ing', 'fucking'), ('fu*king', 'fucking'), ('fuc*ers', 'fuckers'), ('f*****', 'fucking'), ('f***ed', 'fucked'), ('f**ker', 'fucker'), ('f*cked', 'fucked'), ('f*cker', 'fucker'), ('f*ckin', 'fucking'), ('fu*ker', 'fucker'), ('fuc**n', 'fucking'), ('ni**as', 'niggas'), ('b**ch', 'bitch'), ('b*tch', 'bitch'), ('c*unt', 'cunt'), ('f**ks', 'fucks'), ('f*ing', 'fucking'), ('ni**a', 'nigga'), ('c*ck', 'cock'), ('c*nt', 'cunt'), ('cr*p', 'crap'), ('d*ck', 'dick'), ('f***', 'fuck'), ('f**k', 'fuck'), ('f*ck', 'fuck'), ('fc*k', 'fuck'), ('fu**', 'fuck'), ('fu*k', 'fuck'), ('s***', 'shit'), ('s**t', 'shit'), ('sh**', 'shit'), ('sh*t', 'shit'), ('tw*t', 'twat')]

## 替换为后者
fasttext_misspelings = {"'n'balls": 'balls', "-nazi's": 'nazis', 'adminabuse': 'admin abuse', "admins's": 'admins', 'arsewipe': 'arse wipe', 'assfack': 'asshole', 'assholifity': 'asshole', 'assholivity': 'asshole', 'asshoul': 'asshole', 'asssholeee': 'asshole', 'belizeans': 'mexicans', "blowing's": 'blowing', 'bolivians': 'mexicans', 'celtofascists': 'fascists', 'censorshipmeisters': 'censor', 'chileans': 'mexicans', 'clerofascist': 'fascist', 'cowcrap': 'crap', 'crapity': 'crap', "d'idiots": 'idiots', 'deminazi': 'nazi', 'dftt': "don't feed the troll", 'dildohs': 'dildo', 'dramawhores': 'drama whores', 'edophiles': 'pedophiles', 'eurocommunist': 'communist', 'faggotkike': 'faggot', 'fantard': 'retard', 'fascismnazism': 'fascism', 'fascistisized': 'fascist', 'favremother': 'mother', 'fuxxxin': 'fucking', "g'damn": 'goddamn', 'harassmentat': 'harassment', 'harrasingme': 'harassing me', 'herfuc': 'motherfucker', 'hilterism': 'fascism', 'hitlerians': 'nazis', 'hitlerites': 'nazis', 'hubrises': 'pricks', 'idiotizing': 'idiotic', 'inadvandals': 'vandals', "jackass's": 'jackass', 'jiggabo': 'nigga', 'jizzballs': 'jizz balls', 'jmbass': 'dumbass', 'lejittament': 'legitimate', "m'igger": 'nigger', "m'iggers": 'niggers', 'motherfacking': 'motherfucker', 'motherfuckenkiwi': 'motherfucker', 'muthafuggas': 'niggas', 'nazisms': 'nazis', 'netsnipenigger': 'nigger', 'niggercock': 'nigger', 'niggerspic': 'nigger', 'nignog': 'nigga', 'niqqass': 'niggas', "non-nazi's": 'not a nazi', 'panamanians': 'mexicans', 'pedidiots': 'idiots', 'picohitlers': 'hitler', 'pidiots': 'idiots', 'poopia': 'poop', 'poopsies': 'poop', 'presumingly': 'obviously', 'propagandaanddisinformation': 'propaganda and disinformation', 'propagandaministerium': 'propaganda', 'puertoricans': 'mexicans', 'puertorricans': 'mexicans', 'pussiest': 'pussies', 'pussyitis': 'pussy', 'rayaridiculous': 'ridiculous', 'redfascists': 'fascists', 'retardzzzuuufff': 'retard', "revertin'im": 'reverting', 'scumstreona': 'scums', 'southamericans': 'mexicans', 'strasserism': 'fascism', 'stuptarded': 'retarded', "t'nonsense": 'nonsense', "threatt's": 'threat', 'titoists': 'communists', 'twatbags': 'douchebags', 'youbollocks': 'you bollocks'}

## 替换为后者
cont_patterns = [
    (r'(W|w)on\'t', r'will not'),
    (r'(C|c)an\'t', r'can not'),
    (r'(I|i)\'m', r'i am'),
    (r'(A|a)in\'t', r'is not'),
    (r'(\w+)\'ll', r'\g<1> will'),
    (r'(\w+)n\'t', r'\g<1> not'),
    (r'(\w+)\'ve', r'\g<1> have'),
    (r'(\w+)\'s', r'\g<1> is'),
    (r'(\w+)\'re', r'\g<1> are'),
    (r'(\w+)\'d', r'\g<1> would'),
]
