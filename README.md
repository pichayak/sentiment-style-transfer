# sentiment-style-transfer

### Overview
We developed a sentiment style transfer system using a delete, retrieve and generate approach. The final goal is the server-side application that can detect negative sentiment sentences on a chat room and suggest alternative positive sentiment sentences. To address mentioned requirements the system first uses WangchanBerta sentiment classification to detect negative sentences and then pass to the three modules for sentiment style transfer.  The modules to delete, retrieve and generate NLP systems were implemented as followed: Delete module for discover crucial words as attributes which influence the sentence to be negative, Retrieve module for find attributes which used to replace deleted words that made the sentence to be positive and lastly, Generate module for blending retrieved positive attributes into the sentence given context of the sentence. Additionally, we deploy the system on Google Colab using Ngrok for integration with line api. Our sentiment style transfer system can somewhat suggest positive sentences successfully on particular domains that are associated with training dataset.

### before you run AppPy.ipynb (which is webhook for LineAPIChatbot)
- change the lineAccessToken to your own 
- if you can't open in your machine just copy the "appPy" folder to google drive, then open via GoogleColab

### after you run AppPy.ipynb
1. go through your LineChatbot Account (https://developers.line.biz/)
2. select "Messaging API"
3. Eneble Webhook
4. copy url from AppPy.ipynb (url will show at bottom of console if you do everything that I said above)
5. try to talk with your chatbot!



