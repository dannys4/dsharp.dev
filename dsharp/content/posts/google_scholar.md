+++
title = "Google Scholar Daily Digest"
date=2025-01-14
draft=false
+++

I think that Google scholar is a great way of organizing incoming research, and personally use it for keeping track of new research. In particular, I use it for following new papers from particular researchers across my field as well as seeing citations to work of researchers particularly close to me. My one major complaint though is that, if you follow 10 researchers, then it will send you 10 emails every time something happens; particularly frustrating for when you follow who cites a few researchers, as you will be bombarded with emails about different research topics. As such, I tried to fix this problem using Google Apps Script, which is basically just a Google JS-equivalent to VBA for MS Office, as far as I can tell. I used some LLM tools to help me generate this script. Here's what you need to work:

## Setting up the landscape
- First, sign up for whatever Google scholar stuff you'd like to see!
- Then, in your Gmail account, create a new label named `GoogleScholar`
- Open the "Advanced Search" tab and type `scholaralerts-noreply@google.com` into the search bar.
- Instead of pressing search, create a new filter that automatically takes any email from this address and adds the `GoogleScholar` label
- If you want, also make the filter mark the email as read

## Using Google Apps Script
- Log into [https://script.google.com/home](https://script.google.com/home)
- You might be able to use my script [here](https://script.google.com/d/1Oy8Alm1YgDnEcIWZo6kDZ2tygzWF6IGP39NHAnXiZNB0s8Zfehg-BK8c/edit?usp=sharing), but I honestly don't know
- If so, great! Make sure that you authorize access, then skip the rest of these steps
- If not, create a new project and copy the code from the gist [here](https://gist.github.com/dannys4/90c544b6812390e76b9c6b9ff526b875) into the editor
- Add the Gmail service on the left sidebar
- Give authorizations as you wish
- Click `Run` at the top to test that everything works right
    - you may want to label something as `GoogleScholar` just to check; note that it will delete by default
- Click `Triggers` on the fully left sidebar and `+ Add Trigger` on the bottom right
- Choose the `sendDailyEmail` function, `Head` deployment, `Time-driven` event source, and whatever frequency you want (e.g., `Day timer` and `Midnight to 1am` for daily checking)
- Click `Save`.

## That should set you up! Here's the Gist
{{ gist(url="https://gist.github.com/dannys4/90c544b6812390e76b9c6b9ff526b875", class="gist") }}