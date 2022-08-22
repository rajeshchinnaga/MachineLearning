#!/usr/bin/env python
# coding: utf-8

# ### Collecting Datasets:  

# 1) https://www.phishtank.com/developer_info.php - Verified_online_csv file has been taken from phishtank website that contains all the phishing urls.

# 2) https://www.unb.ca/cic/datasets/url-2016.html - All the legitimate URL's has been taken form the mentioned website, Provide the details after clocking the down load dataset. Unzip the folder and select URL and from the URL a file called "Benign_list_big_final.csv"" will be available which contains True URL's.

# **Phishing URLs:**

# In[ ]:


import pandas as pd
phishingDataFrame = pd.read_csv("verified_online.csv")
phishingDataFrame.head()


# In[ ]:


phishingDataFrame.shape


# "Verified_online.csv" file has more than 20,000 records. so among them we are selecting 5000 samples of data for our project purpose.

# In[ ]:


phishingUrl = phishingDataFrame.sample(n = 5000).copy()
phishingUrl = phishingUrl.reset_index(drop=True)


# In[ ]:


phishingUrl.head()


# In[ ]:


phishingUrl.shape


# **True/legitimate URLs:**

# In[ ]:


import pandas as pd
trueUrlDataFrame = pd.read_csv("Benign_list_big_final.csv")
trueUrlDataFrame.head()


# In[ ]:


trueUrlDataFrame.columns = ['url']
trueUrlDataFrame.head()


# In[ ]:


trueUrlDataFrame.shape


# "Benign_list_big_final.csv" file has more than 35,000 records. so among them we are selecting 5000 samples of data for our project purpose.

# In[ ]:


trueUrl = trueUrlDataFrame.sample(n = 5000).copy()
trueUrl = trueUrl.reset_index(drop=True)


# In[ ]:


trueUrl.head()


# In[ ]:


trueUrl.shape


# ### Extracting Features from the Phishing and True URL datasets :

# In[ ]:


from urllib.parse import urlparse,urlencode
import re
import ipaddress


# **Domain of the Url:**

# In[ ]:


def extractDomain(url):  
  dom = urlparse(url).netloc
  if re.match(r"^www.",dom):
      dom = dom.replace("www.","")
  return dom


# **IP Adress from the URL**

# We are marking ip_address as 1 when we find IP address in the URL which means phishing otherwise 0 which means true/Legit URL.

# In[ ]:


def isIP(url):
  try:
    ipaddress.ip_address(url)
    ipAddress = 1
  except:
    ipAddress = 0
  return ipAddress


# **Length of the URL**

# We are marking the length as 1 if the length of the URL is more than 60 which means Phishing URL otherwise 0 which means True/Legit URL.

# In[ ]:


def checkLength(url):
  if len(url) > 60:
    urlLength = 1            
  else:
    urlLength = 0            
  return urlLength


# **Depth of the URL**

# Depth of the url signifies the number of pages present on the website, we can find out that by the number of "/" in the url.

# In[ ]:


def checkDepth(url):
  pages = urlparse(url).path.split('/')
  urlDepth = 0
  for i in range(len(pages)):
    if len(pages[i]) != 0:
      urlDepth = urlDepth+1
  return urlDepth


# **http/https in Domain name**

# If we have http/https in the domain name then we marking the Domain as 1 which is phishing otherwise 0 which means true/legit URL.

# In[ ]:


def checkHttp(url):
  dom = urlparse(url).netloc
  if 'https' in dom:
    return 1
  else:
    return 0


# **@ Symbol**

# We are marking the URL as 1 - phishing if the URL has @ otherwise 0 - if the URL doesnt have @ symbol which mean true/legit URL.

# In[ ]:


def checkSymbol(url):
  if "@" in url:
    symbol = 1    
  else:
    symbol = 0    
  return symbol


# **"-" in Domain**

# We are marking the label as 1 if the URL has - symbol which means phishing otherwise as 0 which means true/legit URL.

# In[ ]:


def checkDashSymbol(url):
    if '-' in urlparse(url).netloc:
        return 1            
    else:
        return 0           


# **// in URL**

# If // is available at 6th or 7th position then we are marking those urls as true (0)  which means legit/true URL's otherwise we will mark as 1 which means phishing.

# In[ ]:


def checkRedirection(url):
  position = url.rfind('//')
  if position != 6 or position!=7:
      return 1
  else:
    return 0


# #### Tiny URL Usage:

# We are marking the URL's that has tiny URL as 1 which means phishiing otherwise 0 which means true/legit URL.

# In[ ]:


tinyUrl = r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"                       r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"                       r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"                       r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|"                       r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|"                       r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|"                       r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"                       r"tr\.im|link\.zip\.net"


# In[ ]:


def checkTinyURL(url):
    match=re.search(tinyUrl,url)
    if match:
        return 1
    else:
        return 0


# ### HTML Features

# In[ ]:


import requests


# #### Website Redirection

# In[ ]:


def redirection(response):
  if response == "":
    return 1
  else:
    if len(response.history) <= 2:
      return 0
    else:
      return 1


# #### Right Click Disabled

# In[ ]:


def checkRightClick(response):
  if response == "":
    return 1
  else:
    if re.findall(r"event.button ?== ?2", response.text):
      return 0
    else:
      return 1


# #### Using IFrame for redirection

# In[ ]:


def checkIframe(response):
  if response == "":
      return 1
  else:
      if re.findall(r"[<iframe>|<frameBorder>]", response.text):
          return 0
      else:
          return 1


# #### Customization of the status bar

# In[ ]:


def mouseOverStatusBar(response): 
  if response == "" :
    return 1
  else:
    if re.findall("<script>.+onmouseover.+</script>", response.text):
      return 1
    else:
      return 0


# ### Additional Features:

# In[ ]:


import re
from bs4 import BeautifulSoup
import whois
import urllib
import urllib.request
from datetime import datetime


# #### Age of Domain

# Since most phishing websites are only active for a few time, we will use the WHOIS database for this. We set the cutoff point at 12, thus all websites older than 12 months will be labeled as phishing while the others will be identified as genuine.

# In[ ]:


def ageOfDomain(domain_name):
  createdDate = domain_name.creation_date
  expirationDate = domain_name.expiration_date
  if (isinstance(createdDate,str) or isinstance(expirationDate,str)):
    try:
      createdDate = datetime.strptime(createdDate,'%Y-%m-%d')
      expirationDate = datetime.strptime(expirationDate,"%Y-%m-%d")
    except:
      return 1
  if ((expirationDate is None) or (createdDate is None)):
      return 1
  elif ((type(expirationDate) is list) or (type(createdDate) is list)):
      return 1
  else:
    domainAge = abs((expirationDate - createdDate).days)
    if ((domainAge/30) < 6):
      age = 1
    else:
      age = 0
  return age


# **End of Domain**

# We'll make use of the WHOIS database. The remaining domain time for this feature is determined by comparing the current time to the expiration time. For this project, the end time taken into account for the legal realm is six months or fewer.
# 
# The value of this feature is 1 (phishing) if the end time of the domain exceeds six months; otherwise, it is 0. (true).

# In[ ]:


def domainEnd(domain_name):
  expiration_date = domain_name.expiration_date
  if isinstance(expiration_date,str):
    try:
      expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
    except:
      return 1
  if (expiration_date is None):
      return 1
  elif (type(expiration_date) is list):
      return 1
  else:
    today = datetime.now()
    end = abs((expiration_date - today).days)
    if ((end/30) < 6):
      end = 0
    else:
      end = 1
  return end


# #### Website Traffic

# A website's popularity can help determine whether it is a legitimate website or a phishing website. We may determine this by counting the amount of webpage hits. Since the Alexa database provides a compilation of the most popular websites worldwide, we will use it for this. A website will be flagged as phishing if its Alexa score is less than 100000; else, it will be marked as 0.

# In[ ]:


def checkWebTrafficPopularity(url):
  try:
    url = urllib.parse.quote(url)
    rank = BeautifulSoup(urllib.request.urlopen("http://data.alexa.com/data?cli=10&dat=s&url=" + url).read(), "xml").find(
        "REACH")['RANK']
    rank = int(rank)
  except TypeError:
        return 1
  if rank <100000:
    return 1
  else:
    return 0


# ### Creating a function for Feature Extraction

# In[ ]:


def featureExtraction(url,label):

  features = []
  features.append(extractDomain(url))
  features.append(isIP(url))
  features.append(checkSymbol(url))
  features.append(checkLength(url))
  features.append(checkDepth(url))
  features.append(checkRedirection(url))
  features.append(checkHttp(url))
  features.append(checkTinyURL(url))
  features.append(checkDashSymbol(url))
  dns = 0
  try:
    domain_name = whois.whois(urlparse(url).netloc)
  except:
    dns = 1
  features.append(dns)
  features.append(checkWebTrafficPopularity(url))
  features.append(1 if dns == 1 else ageOfDomain(domain_name))
  features.append(1 if dns == 1 else domainEnd(domain_name))
  
  # HTML & Javascript based features (4)
  try:
    response = requests.get(url)
  except:
    response = ""
  features.append(checkIframe(response))
  features.append(mouseOverStatusBar(response))
  features.append(checkRightClick(response))
  features.append(redirection(response))
  features.append(label)
  
  return features


# ### Extracting Features of the Phishing URLS

# In[ ]:


from tqdm import tqdm
phishingFeatures = []
label = 1
for i in tqdm(range(0, 5000)):
  url = phishingUrl['url'][i]
  phishingFeatures.append(featureExtraction(url,label))


# In[ ]:


phishingFeatures


# In[ ]:


featureNames = ['Domain', 'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards', 'Label']

phishingDataset = pd.DataFrame(phishingFeatures, columns= featureNames)


# ### Extracting Features of the True URLS

# In[ ]:


from tqdm import tqdm
trueFeatures = []
label = 0

for i in tqdm(range(0, 5000)):
  url = trueUrl['url'][i]
  trueFeatures.append(featureExtraction(url,label))


# In[ ]:


featureNames = ['Domain', 'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth','Redirection', 
                      'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 'Web_Traffic', 
                      'Domain_Age', 'Domain_End', 'iFrame', 'Mouse_Over','Right_Click', 'Web_Forwards', 'Label']

trueDataset = pd.DataFrame(trueFeatures, columns= featureNames)


# In[ ]:


phishingProjectDataset = pd.concat([trueDataset, phishingDataset]).reset_index(drop=True)


# "phishingProjectDataset.csv" - This file contains both 5000 Phishing URL's along with features and 5000 Legit/True URLS's along with features which we will be using for training the models.

# In[ ]:


phishingProjectDataset.to_csv('phishingProjectDataset.csv', index=False)

