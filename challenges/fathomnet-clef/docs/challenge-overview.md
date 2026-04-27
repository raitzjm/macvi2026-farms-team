# FathomNet 2026 — Aggregated Competition Reference

> **Aggregated on 2026-04-27** from the following sources:
> - Kaggle competition Overview page: https://www.kaggle.com/competitions/fathomnet-2026
> - Kaggle competition Data page: https://www.kaggle.com/competitions/fathomnet-2026/data
> - Kaggle competition Rules page: https://www.kaggle.com/competitions/fathomnet-2026/rules
> - ImageCLEF task page: https://www.imageclef.org/FathomNetCLEF2026
> - GitHub README: https://github.com/fathomnet/fgvc-comp-2026
>
> Sources were scraped by Gemini Flash, MiniMax, Claude Sonnet, and Nemotron agents.
> Where sources overlap, the most complete version is used. Unique content from each source is included.
> This document preserves verbatim competition text — it is not a summary or paraphrase.

---

## Overview

**FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC**

Positive-Unlabeled Object Detection in Marine Images

Underwater imaging opens a unique window into the deep sea. It is one of the few ways scientists can directly observe marine animals and habitats, creating visual records that are essential for understanding our changing ocean. But humans generate more ocean data than we can handle, and computer vision is vital to process the thousands of hours of video collected by researchers around the world. Compared to terrestrial datasets, marine imagery presents many unique challenges for computer vision, including high taxonomic diversity, visually similar morphologies, and environments that range from benthic habitats to the open midwater.

One of the biggest challenges? Incomplete labels. Marine biologists are experts in specific animals so they only label what they know in each image. Consequently, annotators may carefully focus on jellyfish in a given image, while ignoring the dozen other creatures swimming by. Just because something isn't labeled doesn't mean it isn't there, and that's a serious problem for anyone trying to teach a computer to recognize all ocean life.

### Description

The FathomNet 2026 challenge addresses this real-world constraint by focusing on positive-unlabeled learning for object detection. Developing robust methods in this setting will reduce reliance on expert annotations, accelerate dataset curation, and broaden the usability of large-scale archives like the FathomNet Database. Ultimately, improving detection under positive-unlabeled regimes will support scalable biodiversity monitoring, enhance conservation strategies, and enable marine researchers to extract ecological insights from imagery collected in the past and into the future.

---

## Data

### Dataset Description

The dataset is derived from the FathomNet Database, which contains more than 400k images and 1M expert-curated bounding boxes, as of Feb 2026, across thousands of marine taxonomic, geologic, and equipment classes. The dataset is formatted to adhere to the COCO Object Detection standard. For the challenge, participants will be provided with a training set of 6,463 images with both labeled and unlabeled positive instances. The test set, by contrast, is composed of 1,425 fully annotated images. To ensure broad accessibility, the entire dataset is less than 50 GB in size.

The FathomNet Database contains a wide spectrum of class labels. For this challenge, these labels were consolidated upstream into 32 grouped categories by our expert team for the FathomVerse game, based on morphological and taxonomic considerations. The dataset includes only annotations using these 32 consolidated categories; the original FathomNet labels and the label mapping are not included. A full list of these categories can be found below.

- amphipod
- anemone
- barnacle
- benthic worm
- bivalve
- black coral
- bony fish
- brittle star
- calycophoran siphonophore
- chiton
- crab
- feather star
- hydroid
- isopod
- jelly
- larvacean
- octopus
- physonect siphonophore
- pyrosome
- sea cucumber
- sea fan
- sea pen
- sea slug
- sea snail
- sea squirt
- sea star
- shrimp
- soft coral
- sponge
- squat lobster
- stony coral
- urchin

### Data Format

The datasets are formatted to adhere to the COCO Object Detection standard. Every training image contains at least one annotation corresponding to 32 distinct categories.
We are not able to provide images as a single downloadable archive due to FathomNet Database's Terms of Use. Images should be downloaded using the indicated coco_url field in each of the annotation files. Participants can either use the provided download.py Python script or write their own. The download script will save one folder containing the images. The image folder contains files with names specified by the file_name field in dataset.json (typically UUID.extension format like 000b8e39-7240-49fd-9f50-713edcb28544.png). The id field is used for COCO format references between images and annotations.

### Files

- `dataset_train.json` — the training images, annotations, and categories in COCO formatted json
- `dataset_test.json` — the evaluation images in COCO formatted json
- `sample_submission.csv` — a sample submission file in the correct format
- `download.py` — python script to download imagery from FathomNet
- `requirements.txt` — Python dependencies for the download script

### Terms of Use

By downloading and using this dataset you are agreeing to FathomNet's data use policy. In particular: The images are licensed under one of three Creative Commons licenses: CC0, CC-BY, CC-BY-NC, and CC-BY-NC-ND. The annotations are licensed under a Creative Commons license CC0, CC-BY or CC-BY-NC. For more details please see the FathomNet Database Terms of Use.

---

## Evaluation

Participants are tasked with developing object detection systems that can operate effectively in a positive-unlabeled learning regime, identifying both the presence and location of marine organisms across diverse habitats. The evaluation metric is the mean Average Precision (mAP), applied to a fully labeled evaluation set. For more information on mAP see here. The evaluation notebook can be found here. To succeed, participants are expected to explore approaches that leverage unlabeled positives, including positive-unlabeled learning, semi-supervised, and self-supervised techniques. The challenge not only serves to benchmark approaches in positive-unlabeled detection but also aims to produce practical methods that can generalize to large-scale, heterogeneous marine datasets, directly supporting ongoing conservation and ecological research.

---

## Submission Format

Submission format should resemble the following:

| annotation_id | image_id | category_id | bbox_x | bbox_y | bbox_width | bbox_height | score |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 28 | 476.0 | 790.0 | 77.0 | 149.0 | 0.3 |

Submissions must be provided as a CSV file where each row corresponds to a single predicted object detection. The file must contain the following columns: annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_width, bbox_height, score.

The annotation_id is a unique (but otherwise arbitrary) identifier for each prediction and is used only as a row index. The image_id must exactly match the image_id of the corresponding image in the COCO-formatted test set. The category_id must match the category IDs used in both the training and test datasets.
Bounding boxes are specified using the COCO object detection format: bbox_x and bbox_y give the pixel coordinates of the top-left corner of the box, while bbox_width and bbox_height specify the width and height of the box in pixels.
The score column represents the model's confidence for the predicted detection. Scores must be floating-point values between 0 and 1 (inclusive). Predictions may be provided in any order.

---

## Rules

### Competition-Specific Rules

ENTRY IN THIS COMPETITION CONSTITUTES YOUR ACCEPTANCE OF THESE OFFICIAL COMPETITION RULES.

See Section 3.18 for defined terms

The Competition named below is a skills-based competition to promote and further the field of data science. You must register via the Competition Website to enter. To enter the Competition, you must agree to these Official Competition Rules, which incorporate by reference the provisions and content of the Competition Website and any Specific Competition Rules herein (collectively, the "Rules"). Please read these Rules carefully before entry to ensure you understand and agree. You further agree that Submission in the Competition constitutes agreement to these Rules. You may not submit to the Competition and are not eligible to receive the prizes associated with this Competition unless you agree to these Rules. These Rules form a binding legal agreement between you and the Competition Sponsor with respect to the Competition. Your competition Submissions must conform to the requirements stated on the Competition Website. Your Submissions will be scored based on the evaluation metric described on the Competition Website. Subject to compliance with the Competition Rules, Prizes, if any, will be awarded to Participants with the best scores, based on the merits of the data science models submitted. See below for the complete Competition Rules.

You cannot sign up to Kaggle from multiple accounts and therefore you cannot enter or submit from multiple accounts.

#### 1. COMPETITION-SPECIFIC TERMS

**1. COMPETITION TITLE**

FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC

**2. COMPETITION SPONSOR**

Monterey Bay Aquarium Research Institute (MBARI), FGVC13, and FathomNet

**3. COMPETITION SPONSOR ADDRESS**

7700 Sandholdt Rd, Moss Landing, CA 95039

**4. COMPETITION WEBSITE**

https://www.kaggle.com/competitions/fathomnet-2026/rules

**5. TOTAL PRIZES AVAILABLE:**

Kudos

**6. WINNER LICENSE TYPE:**

Open Source

**7. DATA ACCESS AND USE:**

Images [CC0, CC-BY, CC-BY-NC, CC BY-NC-ND 4.0] Annotations [CC0, CC-BY, CC BY-NC 4.0] for more detail see here

**8. COMPETITION ACKNOWLEDGEMENTS:**

Laura Chrobak (MBARI), Kevin Barnard (MBARI), Kakani Katija (MBARI)

#### 2. COMPETITION-SPECIFIC RULES

In addition to the provisions of the General Competition Rules below, you understand and agree to these Competition-Specific Rules required by the Competition Sponsor:

**1. TEAM LIMITS**

a. The maximum Team size is ten (10). b. Team mergers are allowed and can be performed by the Team leader. In order to merge, the combined Team must have a total Submission count less than or equal to the maximum allowed as of the Team Merger Deadline. The maximum allowed is the number of Submissions per day multiplied by the number of days the competition has been running.

**2. SUBMISSION LIMITS**

a. You may submit a maximum of ten (10) Submissions per day. b. You may select up to two (2) Final Submissions for judging.

**3. COMPETITION TIMELINE**

a. Competition Timeline dates (including Entry Deadline, Final Submission Deadline, Start Date, and Team Merger Deadline, as applicable) are reflected on the competition's Overview > Timeline page.

**4. COMPETITION DATA**

a. Data Access and Use.

Competition Use and Non-Commercial & Academic Research: You may access and use the Competition Data for non-commercial purposes only, including for participating in the Competition and on Kaggle.com forums, and for academic research and education. The Competition Sponsor reserves the right to disqualify any Participant who uses the Competition Data other than as permitted by the Competition Website and these Rules.

Please acknowledge FathomNet and MBARI when using images for publication or communication purposes. For more details please see the full FathomNet Use Policy.

b. Data Security.

You agree to use reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access to the Competition Data. You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition. You agree to notify Kaggle immediately upon learning of any possible unauthorized transmission of or unauthorized access to the Competition Data and agree to work with Kaggle to rectify any unauthorized transmission or access.

**5. WINNER LICENSE**

a. Under Section 2.8 (Winners Obligations) of the General Rules below, you hereby grant and will grant the Competition Sponsor the following license(s) with respect to your Submission if you are a Competition winner:

Open Source: You hereby license and will license your winning Submission and the source code used to generate the Submission under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

For generally commercially available software that you used to generate your Submission that is not owned by you, but that can be procured by the Competition Sponsor without undue expense, you do not need to grant the license in the preceding Section for that software.

In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s).

b. You may be required by the Sponsor to provide a detailed description of how the winning Submission was generated, to the Competition Sponsor's specifications, as outlined in Section 2.8, Winner's Obligations. This may include a detailed description of methodology, where one must be able to reproduce the approach by reading the description, and includes a detailed explanation of the architecture, preprocessing, loss function, training details, hyper-parameters, etc. The description should also include a link to a code repository with complete and detailed instructions so that the results obtained can be reproduced.

**6. EXTERNAL DATA AND TOOLS**

a. The general rule is that participants should only use the provided training and validation data for training models to classify the test images. We do not want participants crawling the web in search of additional data. Pretrained models may be used to construct the algorithms from publicly available academic datasets (e.g., ImageNet, iNaturalist). Please specify (openly in Discussion section) any and all external data and/or models used for training before uploading results.

**7. ELIGIBILITY**

a. Unless otherwise stated in the Competition-Specific Rules above or prohibited by internal policies of the Competition Entities, employees, interns, contractors, officers and directors of Competition Entities may enter and participate in the Competition, but are not eligible to win any Prizes. "Competition Entities" means the Competition Sponsor, Kaggle Inc., and their respective parent companies, subsidiaries and affiliates. If you are such a Participant from a Competition Entity, you are subject to all applicable internal policies of your employer with respect to your participation.

**8. WINNER'S OBLIGATIONS**

a. As a condition to being awarded a Prize, a Prize winner must fulfill the following obligations:

Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission and associated documentation. The delivered software code should follow these documentation guidelines, must be capable of generating the winning Submission, and contain a description of resources required to build and/or run the executable code successfully. For avoidance of doubt, delivered software code should include training code, inference code, and a description of the required computational environment.

a. To the extent that the final model's software code includes generally commercially available software that is not owned by you, but that can be procured by the Competition Sponsor without undue expense, then instead of delivering the code for that software to the Competition Sponsor, you must identify that software, method for procuring it, and any parameters or other information necessary to replicate the winning Submission; Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner's Submission must still meet the requirements of these Rules, including but not limited to Section 2.5 (Winners License), Section 2.8 (Winners Obligations), and Section 3.14 (Warranty, Indemnity, and Release).

b. Individual Participants and Teams who create a Submission using an AMLT may win a Prize. However, for clarity, the potential winner's Submission must still meet the requirements of these Rules,

Grant to the Competition Sponsor the license to the winning Submission stated in the Competition Specific Rules above, and represent that you have the unrestricted right to grant that license;

Sign and return all Prize acceptance documents as may be required by Competition Sponsor or Kaggle, including without limitation: (a) eligibility certifications; (b) licenses, releases and other agreements required under the Rules; and (c) U.S. tax forms (such as IRS Form W-9 if U.S. resident, IRS Form W-8BEN if foreign resident, or future equivalents).

**9. GOVERNING LAW**

a. Unless otherwise provided in the Competition Specific Rules above, all claims arising out of or relating to these Rules will be governed by California law, excluding its conflict of laws rules, and will be litigated exclusively in the Federal or State courts of Santa Clara County, California, USA. The parties consent to personal jurisdiction in those courts. If any provision of these Rules is held to be invalid or unenforceable, all remaining provisions of the Rules will remain in full force and effect.

---

### Kaggle Competition Foundational Rules

Competition participants must also agree to Kaggle's Foundational Competition Rules. These rules will supersede the competition-specific rules in the event of any conflict.

The following Kaggle Competition Foundational Rules ("Foundational Rules") apply to every competition regardless of whether the Sponsor creates competition-specific rules. Any competition-specific rules provided by the Sponsor are in addition to these rules, and in the case of any conflict or inconsistency, these Foundational Rules control and nullify contrary competition-specific rules.

#### GENERAL COMPETITION RULES — BINDING AGREEMENT

**1. ELIGIBILITY**

a. To be eligible to enter the Competition, you must be:

1. a registered account holder at Kaggle.com;
2. the older of 18 years old or the age of majority in your jurisdiction of residence (unless otherwise agreed to by Competition Sponsor and appropriate parental/guardian consents have been obtained by Competition Sponsor);
3. not a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, or North Korea; and
4. not a person or representative of an entity under U.S. export controls or sanctions (see: https://www.treasury.gov/resourcecenter/sanctions/Programs/Pages/Programs.aspx).

b. Competitions are open to residents of the United States and worldwide, except that if you are a resident of Crimea, so-called Donetsk People's Republic (DNR) or Luhansk People's Republic (LNR), Cuba, Iran, Syria, North Korea, or are subject to U.S. export controls or sanctions, you may not enter the Competition. Other local rules and regulations may apply to you, so please check your local laws to ensure that you are eligible to participate in skills-based competitions. The Competition Host reserves the right to forego or award alternative Prizes where needed to comply with local laws. If a winner is located in a country where prizes cannot be awarded, then they are not eligible to receive a prize.

c. If you are entering as a representative of a company, educational institution or other legal entity, or on behalf of your employer, these rules are binding on you, individually, and the entity you represent or where you are an employee. If you are acting within the scope of your employment, or as an agent of another party, you warrant that such party or your employer has full knowledge of your actions and has consented thereto, including your potential receipt of a Prize. You further warrant that your actions do not violate your employer's or entity's policies and procedures.

d. The Competition Sponsor reserves the right to verify eligibility and to adjudicate on any dispute at any time. If you provide any false information relating to the Competition concerning your identity, residency, mailing address, telephone number, email address, ownership of right, or information required for entering the Competition, you may be immediately disqualified from the Competition.

**2. SPONSOR AND HOSTING PLATFORM**

a. The Competition is sponsored by Competition Sponsor named above. The Competition is hosted on behalf of Competition Sponsor by Kaggle Inc. ("Kaggle"). Kaggle is an independent contractor of Competition Sponsor, and is not a party to this or any agreement between you and Competition Sponsor. You understand that Kaggle has no responsibility with respect to selecting the potential Competition winner(s) or awarding any Prizes. Kaggle will perform certain administrative functions relating to hosting the Competition, and you agree to abide by the provisions relating to Kaggle under these Rules. As a Kaggle.com account holder and user of the Kaggle competition platform, remember you have accepted and are subject to the Kaggle Terms of Service at www.kaggle.com/terms in addition to these Rules.

**3. COMPETITION PERIOD**

a. For the purposes of Prizes, the Competition will run from the Start Date and time to the Final Submission Deadline (such duration the "Competition Period"). The Competition Timeline is subject to change, and Competition Sponsor may introduce additional hurdle deadlines during the Competition Period. Any updated or additional deadlines will be publicized on the Competition Website. It is your responsibility to check the Competition Website regularly to stay informed of any deadline changes. YOU ARE RESPONSIBLE FOR DETERMINING THE CORRESPONDING TIME ZONE IN YOUR LOCATION.

**4. COMPETITION ENTRY**

a. NO PURCHASE NECESSARY TO ENTER OR WIN. To enter the Competition, you must register on the Competition Website prior to the Entry Deadline, and follow the instructions for developing and entering your Submission through the Competition Website. Your Submissions must be made in the manner and format, and in compliance with all other requirements, stated on the Competition Website (the "Requirements"). Submissions must be received before any Submission deadlines stated on the Competition Website. Submissions not received by the stated deadlines will not be eligible to receive a Prize.

b. Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.

c. If the Competition is a multi-stage competition with temporally separate training and/or test data, one or more valid Submissions may be required during each Competition stage in the manner described on the Competition Website in order for the Submissions to be Prize eligible.

d. Submissions are void if they are in whole or part illegible, incomplete, damaged, altered, counterfeit, obtained through fraud, or late. Competition Sponsor reserves the right to disqualify any entrant who does not follow these Rules, including making a Submission that does not meet the Requirements.

**5. INDIVIDUALS AND TEAMS**

a. Individual Account. You may make Submissions only under one, unique Kaggle.com account. You will be disqualified if you make Submissions through more than one Kaggle account, or attempt to falsify an account to act as your proxy. You may submit up to the maximum number of Submissions per day as specified on the Competition Website.

b. Teams. If permitted under the Competition Website guidelines, multiple individuals may collaborate as a Team; however, you may join or form only one Team. Each Team member must be a single individual with a separate Kaggle account. You must register individually for the Competition before joining a Team. You must confirm your Team membership to make it official by responding to the Team notification message sent to your Kaggle account. Team membership may not exceed the Maximum Team Size stated on the Competition Website.

c. Team Merger. Teams may request to merge via the Competition Website. Team mergers may be allowed provided that: (i) the combined Team does not exceed the Maximum Team Size; (ii) the number of Submissions made by the merging Teams does not exceed the number of Submissions permissible for one Team at the date of the merger request; (iii) the merger is completed before the earlier of: any merger deadline or the Competition deadline; and (iv) the proposed combined Team otherwise meets all the requirements of these Rules.

d. Private Sharing. No private sharing outside of Teams. Privately sharing code or data outside of Teams is not permitted. It's okay to share code if made available to all Participants on the forums.

**6. SUBMISSION CODE REQUIREMENTS**

a. Private Code Sharing. Unless otherwise specifically permitted under the Competition Website or Competition Specific Rules above, during the Competition Period, you are not allowed to privately share source or executable code developed in connection with or based upon the Competition Data or other source or executable code relevant to the Competition ("Competition Code"). This prohibition includes sharing Competition Code between separate Teams, unless a Team merger occurs. Any such sharing of Competition Code is a breach of these Competition Rules and may result in disqualification.

b. Public Code Sharing. You are permitted to publicly share Competition Code, provided that such public sharing does not violate the intellectual property rights of any third party. If you do choose to share Competition Code or other such code, you are required to share it on Kaggle.com on the discussion forum or notebooks associated specifically with the Competition for the benefit of all competitors. By so sharing, you are deemed to have licensed the shared code under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such Competition Code or model containing or depending on such Competition Code.

c. Use of Open Source. Unless otherwise stated in the Specific Competition Rules above, if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license (see www.opensource.org) that in no event limits commercial use of such code or model containing or depending on such code.

**7. DETERMINING WINNERS**

a. Each Submission will be scored and ranked by the evaluation metric stated on the Competition Website. During the Competition Period, the current ranking will be visible on the Competition Website's Public Leaderboard. The potential winner(s) are determined solely by the leaderboard ranking on the Private Leaderboard, subject to compliance with these Rules. The Public Leaderboard will be based on the public test set and the Private Leaderboard will be based on the private test set.

b. In the event of a tie, the Submission that was entered first to the Competition will be the winner. In the event a potential winner is disqualified for any reason, the Submission that received the next highest score rank will be chosen as the potential winner.

**8. NOTIFICATION OF WINNERS & DISQUALIFICATION**

a. The potential winner(s) will be notified by email.

b. If a potential winner (i) does not respond to the notification attempt within one (1) week from the first notification attempt or (ii) notifies Kaggle within one week after the Final Submission Deadline that the potential winner does not want to be nominated as a winner or does not want to receive a Prize, then, in each case (i) and (ii) such potential winner will not receive any Prize, and an alternate potential winner will be selected from among all eligible entries received based on the Competition's judging criteria.

c. In case (i) and (ii) above Kaggle may disqualify the Participant. However, in case (ii) above, if requested by Kaggle, such potential winner may provide code and documentation to verify the Participant's compliance with these Rules. If the potential winner provides code and documentation to the satisfaction of Kaggle, the Participant will not be disqualified pursuant to this paragraph.

d. Competition Sponsor reserves the right to disqualify any Participant from the Competition if the Competition Sponsor reasonably believes that the Participant has attempted to undermine the legitimate operation of the Competition by cheating, deception, or other unfair playing practices or abuses, threatens or harasses any other Participants, Competition Sponsor or Kaggle.

e. A disqualified Participant may be removed from the Competition leaderboard, at Kaggle's sole discretion. If a Participant is removed from the Competition Leaderboard, additional winning features associated with the Kaggle competition platform, for example Kaggle points or medals, may also not be awarded.

f. The final leaderboard list will be publicly displayed at Kaggle.com. Determinations of Competition Sponsor are final and binding.

**9. PRIZES**

a. Prize(s) are as described on the Competition Website and are only available for winning during the time period described on the Competition Website. The odds of winning any Prize depends on the number of eligible Submissions received during the Competition Period and the skill of the Participants.

b. All Prizes are subject to Competition Sponsor's review and verification of the Participant's eligibility and compliance with these Rules, and the compliance of the winning Submissions with the Submissions Requirements. In the event that the Submission demonstrates non-compliance with these Competition Rules, Competition Sponsor may at its discretion take either of the following actions: (i) disqualify the Submission(s); or (ii) require the potential winner to remediate within one week after notice all issues identified in the Submission(s) (including, without limitation, the resolution of license conflicts, the fulfillment of all obligations required by software licenses, and the removal of any software that violates the software restrictions).

c. A potential winner may decline to be nominated as a Competition winner in accordance with Section 3.8.

d. Potential winners must return all required Prize acceptance documents within two (2) weeks following notification of such required documents, or such potential winner will be deemed to have forfeited the prize and another potential winner will be selected. Prize(s) will be awarded within approximately thirty (30) days after receipt by Competition Sponsor or Kaggle of the required Prize acceptance documents. Transfer or assignment of a Prize is not allowed.

e. You are not eligible to receive any Prize if you do not meet the Eligibility requirements in Section 2.7 and Section 3.1 above.

f. If a Team wins a monetary Prize, the Prize money will be allocated in even shares between the eligible Team members, unless the Team unanimously opts for a different Prize split and notifies Kaggle before Prizes are issued.

**10. TAXES**

a. ALL TAXES IMPOSED ON PRIZES ARE THE SOLE RESPONSIBILITY OF THE WINNERS. Payments to potential winners are subject to the express requirement that they submit all documentation requested by Competition Sponsor or Kaggle for compliance with applicable state, federal, local and foreign (including provincial) tax reporting and withholding requirements. Prizes will be net of any taxes that Competition Sponsor is required by law to withhold. If a potential winner fails to provide any required documentation or comply with applicable laws, the Prize may be forfeited and Competition Sponsor may select an alternative potential winner. Any winners who are U.S. residents will receive an IRS Form-1099 in the amount of their Prize.

**11. GENERAL CONDITIONS**

a. All federal, state, provincial and local laws and regulations apply.

**12. PUBLICITY**

a. You agree that Competition Sponsor, Kaggle and its affiliates may use your name and likeness for advertising and promotional purposes without additional compensation, unless prohibited by law.

**13. PRIVACY**

a. You acknowledge and agree that Competition Sponsor and Kaggle may collect, store, share and otherwise use personally identifiable information provided by you during the Kaggle account registration process and the Competition, including but not limited to, name, mailing address, phone number, and email address ("Personal Information"). Kaggle acts as an independent controller with regard to its collection, storage, sharing, and other use of this Personal Information, and will use this Personal Information in accordance with its Privacy Policy <www.kaggle.com/privacy>, including for administering the Competition. As a Kaggle.com account holder, you have the right to request access to, review, rectification, portability or deletion of any personal data held by Kaggle about you by logging into your account and/or contacting Kaggle Support at <www.kaggle.com/contact>.

b. As part of Competition Sponsor performing this contract between you and the Competition Sponsor, Kaggle will transfer your Personal Information to Competition Sponsor, which acts as an independent controller with regard to this Personal Information. As a controller of such Personal Information, Competition Sponsor agrees to comply with all U.S. and foreign data protection obligations with regard to your Personal Information. Kaggle will transfer your Personal Information to Competition Sponsor in the country specified in the Competition Sponsor Address listed above, which may be a country outside the country of your residence. Such country may not have privacy laws and regulations similar to those of the country of your residence.

**14. WARRANTY, INDEMNITY AND RELEASE**

a. You warrant that your Submission is your own original work and, as such, you are the sole and exclusive owner and rights holder of the Submission, and you have the right to make the Submission and grant all required licenses. You agree not to make any Submission that: (i) infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; or (ii) otherwise violates any applicable U.S. or foreign state or federal law.

b. To the maximum extent permitted by law, you indemnify and agree to keep indemnified Competition Entities at all times from and against any liability, claims, demands, losses, damages, costs and expenses resulting from any of your acts, defaults or omissions and/or a breach of any warranty set forth herein. To the maximum extent permitted by law, you agree to defend, indemnify and hold harmless the Competition Entities from and against any and all claims, actions, suits or proceedings, as well as any and all losses, liabilities, damages, costs and expenses (including reasonable attorneys fees) arising out of or accruing from: (a) your Submission or other material uploaded or otherwise provided by you that infringes any third party proprietary rights, intellectual property rights, industrial property rights, personal or moral rights or any other rights, including without limitation, copyright, trademark, patent, trade secret, privacy, publicity or confidentiality obligations, or defames any person; (b) any misrepresentation made by you in connection with the Competition; (c) any non-compliance by you with these Rules or any applicable U.S. or foreign state or federal law; (d) claims brought by persons or entities other than the parties to these Rules arising from or related to your involvement with the Competition; and (e) your acceptance, possession, misuse or use of any Prize, or your participation in the Competition and any Competition-related activity.

c. You hereby release Competition Entities from any liability associated with: (a) any malfunction or other problem with the Competition Website; (b) any error in the collection, processing, or retention of any Submission; or (c) any typographical or other error in the printing, offering or announcement of any Prize or winners.

**15. INTERNET**

a. Competition Entities are not responsible for any malfunction of the Competition Website or any late, lost, damaged, misdirected, incomplete, illegible, undeliverable, or destroyed Submissions or entry materials due to system errors, failed, incomplete or garbled computer or other telecommunication transmission malfunctions, hardware or software failures of any kind, lost or unavailable network connections, typographical or system/human errors and failures, technical malfunction(s) of any telephone network or lines, cable connections, satellite transmissions, servers or providers, or computer equipment, traffic congestion on the Internet or at the Competition Website, or any combination thereof, which may limit a Participant's ability to participate.

**16. RIGHT TO CANCEL, MODIFY OR DISQUALIFY**

a. If for any reason the Competition is not capable of running as planned, including infection by computer virus, bugs, tampering, unauthorized intervention, fraud, technical failures, or any other causes which corrupt or affect the administration, security, fairness, integrity, or proper conduct of the Competition, Competition Sponsor reserves the right to cancel, terminate, modify or suspend the Competition. Competition Sponsor further reserves the right to disqualify any Participant who tampers with the submission process or any other part of the Competition or Competition Website. Any attempt by a Participant to deliberately damage any website, including the Competition Website, or undermine the legitimate operation of the Competition is a violation of criminal and civil laws. Should such an attempt be made, Competition Sponsor and Kaggle each reserves the right to seek damages from any such Participant to the fullest extent of the applicable law.

**17. NOT AN OFFER OR CONTRACT OF EMPLOYMENT**

a. Under no circumstances will the entry of a Submission, the awarding of a Prize, or anything in these Rules be construed as an offer or contract of employment with Competition Sponsor or any of the Competition Entities. You acknowledge that you have submitted your Submission voluntarily and not in confidence or in trust. You acknowledge that no confidential, fiduciary, agency, employment or other similar relationship is created between you and Competition Sponsor or any of the Competition Entities by your acceptance of these Rules or your entry of your Submission.

**18. DEFINITIONS**

a. "Competition Data" are the data or datasets available from the Competition Website for the purpose of use in the Competition, including any prototype or executable code provided on the Competition Website. The Competition Data will contain private and public test sets. Which data belongs to which set will not be made available to Participants.

b. An "Entry" is when a Participant has joined, signed up, or accepted the rules of a competition. Entry is required to make a Submission to a competition.

c. A "Final Submission" is the Submission selected by the user, or automatically selected by Kaggle in the event not selected by the user, that is/are used for final placement on the competition leaderboard.

d. A "Participant" or "Participant User" is an individual who participates in a competition by entering the competition and making a Submission.

e. The "Private Leaderboard" is a ranked display of Participants' Submission scores against the private test set. The Private Leaderboard determines the final standing in the competition.

f. The "Public Leaderboard" is a ranked display of Participants' Submission scores against a representative sample of the test data. This leaderboard is visible throughout the competition.

g. A "Sponsor" is responsible for hosting the competition, which includes but is not limited to providing the data for the competition, determining winners, and enforcing competition rules.

h. A "Submission" is anything provided by the Participant to the Sponsor to be evaluated for competition purposes and determine leaderboard position. A Submission may be made as a model, notebook, prediction file, or other format as determined by the Sponsor.

i. A "Team" is one or more Participants participating together in a Kaggle competition, by officially merging together as a Team within the competition platform.

---

## Guidelines

Participants should not use other image sets to train their models outside of the FathomNet-CLEF Kaggle competition data and the FathomNet Database website, however they can leverage other types of outside data sources for other aspects of the competition. Pre-trained models can be used for initialization of training. Teams should plan on specifying additional data sources and/or pre-trained models used when uploading results.

---

## Timeline

- 17 November 2025: Registration opens for all LifeCLEF challenges (registration is free of charge)
- 6 February – 7 March 2026: Competition Start (depending on the competition)
- 23 April 2026: Registration closes for all LifeCLEF challenges
- 7 May 2026: Kaggle Competition Deadline
- 28 May 2026: Deadline for submission of working note papers by participants [CEUR-WS proceedings]
- 30 June 2026: Notification of acceptance of working note papers [CEUR-WS proceedings]
- 6 July 2026: Camera-ready deadline for working note papers.
- 21-24 Sept 2026: CLEF 2026 Jena - Germany

All deadlines are at 11:59 PM CET of the corresponding day unless otherwise stated. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

---

## CVPR & CLEF Context

This competition is held jointly as part of:
The LifeCLEF 2026 Lab of the CLEF 2026 conference, and of the FGVC13 Workshop organized in conjunction with the CVPR 2026 conference. Being part of scientific research, the participants are encouraged to participate in both events. In particular, only participants who submit a working note paper to LifeCLEF (see below) will be part of the officially published ranking used for scientific communication.

This competition is part of the Fine-Grained Visual Categorization FGVC13 workshop at the Computer Vision and Pattern Recognition (CVPR) Conference 2026 in Denver, Colorado. The task results will be presented at the workshop, and the contribution of the winning team(s) will be highlighted. Attending the workshop is not required to participate in the competition. We highly encourage teams to share a public repository with their submission to aid the community in learning from one another's approach. There is no cash prize for this competition. PLEASE NOTE: CVPR frequently sells out early, and we cannot guarantee CVPR registration after the competition's end. If you are interested in attending CVPR, please plan ahead.

You can see a list of all FGVC13 competitions here.

LifeCLEF 2026 at CLEF 2026 LifeCLEF lab is part of the Conference and Labs of the Evaluation Forum (CLEF). CLEF consists of independent peer-reviewed workshops on a broad range of challenges in multilingual and multimodal information access evaluation and benchmarking activities in various labs designed to test different aspects of mono and cross-language Information retrieval systems. CLEF 2026 will take place in Jena, Germany, on September 21-24, 2026. You can find more details on the CLEF 2026 website.

Please feel free to open an issue on the competition git repo if you have a question, comment, or problem. We will also respond to threads opened on the competition discussion board here on Kaggle.

### Participation Requirements (ImageCLEF page)

1. Subscribe to CLEF (FathomNetCLEF task) by filling this form
2. Go to the Kaggle competition page and compete

---

## Acknowledgements

The images and annotations for this competition have been generously provided by FathomNet, MBARI, the Schmidt Ocean Institute, Dawn Wright, NOAA, Ocean Networks Canada, Joost Daniels. Special acknowledgment is given to Marine Biologist Linda Kuhnz for her invaluable expertise in generating a substantial portion of the ground truth dataset.

---

## Citation

Kevin Barnard and Laura Chrobak. FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC. https://kaggle.com/competitions/fathomnet-2026, 2026. Kaggle.

*(ImageCLEF page lists the citation as "Unpublished" — the Kaggle page lists year as 2026.)*

---

## GitHub README (fathomnet/fgvc-comp-2026)

Source: https://github.com/fathomnet/fgvc-comp-2026

---

# FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC

Positive-Unlabeled Object Detection in Marine Images

## Timeline
- 17 November 2025: Registration opens for all LifeCLEF challenges (registration is free of charge)
- 6 February – 7 March 2026: Competition Start (depending on the competition)
- 23 April 2026: Registration closes for all LifeCLEF challenges
- 7 May 2026: Kaggle Competition Deadline
- 28 May 2026: Deadline for submission of working note papers by participants [CEUR-WS proceedings]
- 30 June 2026: Notification of acceptance of working note papers [CEUR-WS proceedings]
- 6 July 2026: Camera-ready deadline for working note papers.
- 21-24 Sept 2026: CLEF 2026 Jena - Germany
All deadlines are at 11:59 PM CET of the corresponding day unless otherwise stated. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

# Overview

Underwater imaging opens a unique window into the deep sea. It is one of the few ways scientists can directly observe marine animals and habitats, creating visual records that are essential for understanding our changing ocean. But humans generate more ocean data than we can handle, and computer vision is vital to process the thousands of hours of video collected by researchers around the world. Compared to terrestrial datasets, marine imagery presents many unique challenges for computer vision, including high taxonomic diversity, visually similar morphologies, and environments that range from benthic habitats to the open midwater.

One of the biggest challenges? Incomplete labels. Marine biologists are experts in specific animals so they only label what they know in each image. Consequently, annotators may carefully focus on jellyfish in a given image, while ignoring the dozen other creatures swimming by. Just because something isn't labeled doesn't mean it isn't there, and that's a serious problem for anyone trying to teach a computer to recognize all ocean life.

See our kaggle competition page [here](https://www.kaggle.com/competitions/fathomnet-2026).

# Description
The FathomNet 2026 challenge addresses this real-world constraint by focusing on positive-unlabeled learning for object detection. Developing robust methods in this setting will reduce reliance on expert annotations, accelerate dataset curation, and broaden the usability of large-scale archives like the [FathomNet Database](https://database.fathomnet.org/fathomnet/#/). Ultimately, improving detection under positive-unlabeled regimes will support scalable biodiversity monitoring, enhance conservation strategies, and enable marine researchers to extract ecological insights from decades of imagery.

# Evaluation

Participants are tasked with developing object detection systems that can operate effectively in a positive-unlabeled learning regime, identifying both the presence and location of marine organisms across diverse habitats. The evaluation metric is the mean Average Precision (mAP), applied to the fully labeled evaluation set. For more information on mAP see [here](https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/). The evaluation notebook can be found [here](https://www.kaggle.com/code/lauravchrobak/map50-95). To succeed, participants are expected to explore approaches that leverage unlabeled positives, including positive-unlabeled learning, semi-supervised, and self-supervised techniques. The challenge not only serves to benchmark approaches in positive-unlabeled detection but also aims to produce practical methods that can generalize to large-scale, heterogeneous marine datasets, directly supporting ongoing conservation and ecological research.

# Submission Format

Submission format should resemble the following:

| annotation_id | image_id | category_id | bbox_x | bbox_y | bbox_width | bbox_height | score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 28 | 476.0 | 790.0 | 77.0 | 149.0 | 0.3 |

Submissions must be provided as a CSV file where each row corresponds to a single predicted object detection. The file must contain the following columns:
annotation_id, image_id, category_id, bbox_x, bbox_y, bbox_width, bbox_height, score.
The annotation_id is a unique (but otherwise arbitrary) identifier for each prediction and is used only as a row index. The image_id must exactly match the image_id of the corresponding image in the COCO-formatted test set. The category_id must match the category IDs used in both the training and test datasets.
Bounding boxes are specified using the COCO object detection format: bbox_x and bbox_y give the pixel coordinates of the top-left corner of the box, while bbox_width and bbox_height specify the width and height of the box in pixels.
The score column represents the model's confidence for the predicted detection. Scores must be floating-point values between 0 and 1 (inclusive). Predictions may be provided in any order.

## Competition Information

This competition is held jointly as part of:
The [LifeCLEF 2026 Lab](https://www.imageclef.org/LifeCLEF2026) of the CLEF 2026 conference, and of the [FGVC13 Workshop](https://sites.google.com/view/fgvc13/home) organized in conjunction with the CVPR 2026 conference. Being part of scientific research, the participants are encouraged to participate in both events. In particular, only participants who submit a working note paper to LifeCLEF (see below) will be part of the officially published ranking used for scientific communication.

This competition is part of the Fine-Grained Visual Categorization FGVC13 workshop at the Computer Vision and Pattern Recognition (CVPR) Conference 2026 in Denver, Colorado. The task results will be presented at the workshop, and the contribution of the winning team(s) will be highlighted. Attending the workshop is not required to participate in the competition. We highly encourage teams to share a public repository with their submission to aid the community in learning from one another's approach. There is no cash prize for this competition. PLEASE NOTE: CVPR frequently sells out early, and we cannot guarantee CVPR registration after the competition's end. If you are interested in attending CVPR, please plan ahead.

You can see a list of all FGVC13 competitions [here](https://sites.google.com/view/fgvc13/competitions?authuser=0).

LifeCLEF 2026 at CLEF 2026 LifeCLEF lab is part of the Conference and Labs of the Evaluation Forum (CLEF). CLEF consists of independent peer-reviewed workshops on a broad range of challenges in multilingual and multimodal information access evaluation and benchmarking activities in various labs designed to test different aspects of mono and cross-language Information retrieval systems. CLEF 2026 will take place in Jena, Germany, on September 21-24, 2026. You can find more details on the CLEF 2026 [website](https://clef2026.clef-initiative.eu/).

Please feel free to open an issue on the competition git repo if you have a question, comment, or problem. We will also respond to threads opened on the competition discussion board here on Kaggle.

## Acknowledgments

The images and annotations for this competition have been generously provided by FathomNet, MBARI, the Schmidt Ocean Institute, Dawn Wright, NOAA, Ocean Networks Canada, Joost Daniels. Special acknowledgment is given to Marine Biologist Linda Kuhnz for her invaluable expertise in generating a substantial portion of the ground truth dataset.

# Citation
Kevin Barnard and Laura Chrobak. FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC. https://kaggle.com/competitions/fathomnet-2026, Unpublished. Kaggle.

# Data
The dataset is derived from the [FathomNet Database](https://database.fathomnet.org/fathomnet/#/), which contains more than 400k images and 1M expert-curated bounding boxes, as of Feb 2026, across thousands of marine taxonomic, geologic, and equipment classes. The dataset is formatted to adhere to the [COCO Object Detection](https://cocodataset.org/#format-data) standard. For the challenge, participants will be provided with a training set of 6,463 images with both labeled and unlabeled positive instances. The test set, by contrast, is composed of 1,425 fully annotated images. To ensure broad accessibility, the entire dataset is less than 50 GB in size.

The FathomNet Database contains a wide spectrum of class labels. To make this challenge more manageable, we have mapped the labels as they appear in the FathomNet Database to 32 consolidated groupings. These groupings were assembled by our expert team for the [FathomVerse](https://www.fathomverse.game/) game and reflect a combination of morphological and taxonomic categories. A full list of these categories can be found below.
- amphipod
- anemone
- barnacle
- benthic worm
- bivalve
- black coral
- bony fish
- brittle star
- calycophoran siphonophore
- chiton
- crab
- feather star
- hydroid
- isopod
- jelly
- larvacean
- octopus
- physonect siphonophore
- pyrosome
- sea cucumber
- sea fan
- sea pen
- sea slug
- sea snail
- sea squirt
- sea star
- shrimp
- soft coral
- sponge
- squat lobster
- stony coral
- urchin

## Data Format
The datasets are formatted to adhere to the COCO Object Detection standard. Every training image contains at least one annotation corresponding to 32 distinct categories.
We are not able to provide images as a single downloadable archive due to FathomNet Database's Terms of Use. Images should be downloaded using the indicated coco_url field in each of the annotation files. Participants can either use the provided download.py Python script or write their own. The download script will save one folder containing the images. The image folder contains files with names specified by the file_name field in dataset.json (typically UUID.extension format like 000b8e39-7240-49fd-9f50-713edcb28544.png). The id field is used for COCO format references between images and annotations.

Files:
- `dataset_train.json` — the training images, annotations, and categories in COCO formatted json
- `dataset_test.json` — the evaluation images in COCO formatted json
- `sample_submission.csv` — a sample submission file in the correct format
- `download.py` — python script to download imagery from FathomNet

## Guidelines
Participants should not use other image sets to train their models outside of the FathomNet-CLEF Kaggle competition data and the FathomNet Database website, however they can leverage other types of outside data sources for other aspects of the competition. Pre-trained models can be used for initialization of training. Teams should plan on specifying additional data sources and/or pre-trained models used when uploading results.

## Terms of Use
By downloading and using this dataset you are agreeing to FathomNet's [data use](https://www.fathomnet.org/datause) policy. In particular: The images are licensed under one of three Creative Commons licenses: [CC0](https://creativecommons.org/public-domain/cc0/), [CC-BY](https://creativecommons.org/licenses/by/4.0/deed.en), [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.en), and [CC-BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en). The annotations are licensed under a Creative Commons license CC0, CC-BY or CC-BY-NC. For more details please see the FathomNet Database [Terms of Use](https://www.fathomnet.org/terms).

## Download

This directory contains the code and requirements for downloading the training and test datasets used for FathomNetCLEF 2026.

### Requirements

- Python 3.10+
- Packages listed in `requirements.txt` (available via PyPI)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
python download.py [-h] [-o OUTPUT_DIR] [--min-workers MIN_WORKERS] [--max-workers MAX_WORKERS] [--initial-workers INITIAL_WORKERS] dataset_path
```

The script will autoscale the number of workers based on server failures.

---

## ImageCLEF Page (https://www.imageclef.org/FathomNetCLEF2026)

FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC — Positive-Unlabeled Object Detection in Marine Images

### Schedule

- 17 November 2025: Registration opens for all LifeCLEF challenges (registration is free of charge)
- 7 February 2026: Competition Start
- 23 April 2026: Registration closes for all LifeCLEF challenges
- 7 May 2026: Competition Deadline
- 28 May 2026: Deadline for submission of working note papers by participants [CEUR-WS proceedings]
- 30 June 2026: Notification of acceptance of working note papers [CEUR-WS proceedings]
- 6 July 2026: Camera-ready deadline for working note papers.
- 21-24 Sept 2026: CLEF 2026 Jena - Germany

All deadlines are at 11:59 PM CET on a corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

### Overview

Underwater imaging opens a unique window into the deep sea. It is one of the few ways scientists can directly observe marine animals and habitats, creating visual records that are essential for understanding our changing ocean. But humans generate more ocean data than we can handle, and computer vision is vital to process the thousands of hours of video collected by researchers around the world. Compared to terrestrial datasets, marine imagery presents many unique challenges for computer vision, including high taxonomic diversity, visually similar morphologies, and environments that range from benthic habitats to the open midwater.

One of the biggest challenges? Incomplete labels. Marine biologists are experts in specific animals so they only label what they know in each image. Consequently, annotators may carefully focus on jellyfish in a given image, while ignoring the dozen other creatures swimming by. Just because something isn't labeled doesn't mean it isn't there, and that's a serious problem for anyone trying to teach a computer to recognize all ocean life.

See our kaggle competition page here.

### Description

The FathomNet 2026 challenge addresses this real-world constraint by focusing on positive-unlabeled learning for object detection. Developing robust methods in this setting will reduce reliance on expert annotations, accelerate dataset curation, and broaden the usability of large-scale archives like the FathomNet Database. Ultimately, improving detection under positive-unlabeled regimes will support scalable biodiversity monitoring, enhance conservation strategies, and enable marine researchers to extract ecological insights from decades of imagery.

### Evaluation

Participants are tasked with developing object detection systems that can operate effectively in a positive-unlabeled learning regime, identifying both the presence and location of marine organisms across diverse habitats. The evaluation metric is the mean Average Precision (mAP), applied to the fully labeled evaluation set. For more information on mAP see here. The evaluation notebook can be found here. To succeed, participants are expected to explore approaches that leverage unlabeled positives, including positive-unlabeled learning, semi-supervised, and self-supervised techniques. The challenge not only serves to benchmark approaches in positive-unlabeled detection but also aims to produce practical methods that can generalize to large-scale, heterogeneous marine datasets, directly supporting ongoing conservation and ecological research.

### Competition Information

This competition is held jointly as part of: The LifeCLEF 2026 Lab of the CLEF 2026 conference, and of the FGVC13 Workshop organized in conjunction with the CVPR 2026 conference. Being part of scientific research, the participants are encouraged to participate in both events. In particular, only participants who submit a working note paper to LifeCLEF (see below) will be part of the officially published ranking used for scientific communication.

This competition is part of the Fine-Grained Visual Categorization FGVC13 workshop at the Computer Vision and Pattern Recognition (CVPR) Conference 2026 in Denver, Colorado. The task results will be presented at the workshop, and the contribution of the winning team(s) will be highlighted. Attending the workshop is not required to participate in the competition. We highly encourage teams to share a public repository with their submission to aid the community in learning from one another's approach. There is no cash prize for this competition.

PLEASE NOTE: CVPR frequently sells out early, and we cannot guarantee CVPR registration after the competition's end. If you are interested in attending CVPR, please plan ahead. You can see a list of all FGVC13 competitions here.

### LifeCLEF 2026 at CLEF 2026

LifeCLEF lab is part of the Conference and Labs of the Evaluation Forum (CLEF). CLEF consists of independent peer-reviewed workshops on a broad range of challenges in multilingual and multimodal information access evaluation and benchmarking activities in various labs designed to test different aspects of mono and cross-language Information retrieval systems. CLEF 2026 will take place in Jena, Germany, on September 21-24, 2026. You can find more details on the CLEF 2026 website.

Please feel free to open an issue on the competition git repo if you have a question, comment, or problem. We will also respond to threads opened on the competition discussion board here on Kaggle.

### Participation Requirements

1. Subscribe to CLEF (FathomNetCLEF task) by filling this form
2. Go to the Kaggle competition page and compete

### Acknowledgments

The images and annotations for this competition have been generously provided by FathomNet, MBARI, the Schmidt Ocean Institute, Dawn Wright, NOAA, Ocean Networks Canada, Joost Daniels. Special acknowledgment is given to Marine Biologist Linda Kuhnz for her invaluable expertise in generating a substantial portion of the ground truth dataset.

### Citation

Kevin Barnard and Laura Chrobak. FathomNetCLEF2026 @ LifeCLEF & CVPR-FGVC. https://kaggle.com/competitions/fathomnet-2026, Unpublished. Kaggle.

### Data

The dataset is derived from the FathomNet Database, which contains more than 400k images and 1M expert-curated bounding boxes, as of Feb 2026, across thousands of marine taxonomic, geologic, and equipment classes. The dataset is formatted to adhere to the COCO Object Detection standard. For the challenge, participants will be provided with a training set of 6,463 images with both labeled and unlabeled positive instances. The test set, by contrast, is composed of 1,425 fully annotated images. To ensure broad accessibility, the entire dataset is less than 50 GB in size.

The FathomNet Database contains a wide spectrum of class labels. To make this challenge more manageable, we have mapped the labels as they appear in the FathomNet Database to 32 consolidated groupings. These groupings were assembled by our expert team for the FathomVerse game and reflect a combination of morphological and taxonomic categories. A full list of these categories can be found below:

- amphipod
- anemone
- barnacle
- benthic worm
- bivalve
- black coral
- bony fish
- brittle star
- calycophoran siphonophore
- chiton
- crab
- feather star
- hydroid
- isopod
- jelly
- larvacean
- octopus
- physonect siphonophore
- pyrosome
- sea cucumber
- sea fan
- sea pen
- sea slug
- sea snail
- sea squirt
- sea star
- shrimp
- soft coral
- sponge
- squat lobster
- stony coral
- urchin

### Data Format

The datasets are formatted to adhere to the COCO Object Detection standard. Every training image contains at least one annotation corresponding to 32 distinct categories. We are not able to provide images as a single downloadable archive due to FathomNet Database's Terms of Use. Images should be downloaded using the indicated coco_url field in each of the annotation files. Participants can either use the provided download.py Python script or write their own. The download script will save one folder containing the images. The image folder contains files with names specified by the file_name field in dataset.json (typically UUID.extension format like 000b8e39-7240-49fd-9f50-713edcb28544.png). The id field is used for COCO format references between images and annotations.

Files:
- `dataset_train.json` — the training images, annotations, and categories in COCO formatted json
- `dataset_test.json` — the evaluation images in COCO formatted json
- `sample_submission.csv` — a sample submission file in the correct format
- `download.py` — python script to download imagery from FathomNet

### Terms of Use

By downloading and using this dataset you are agreeing to FathomNet's data use policy. In particular: The images are licensed under one of three Creative Commons licenses: CC0, CC-BY, and CC-BY-NC-ND. The annotations are licensed under a Creative Commons license CC0, CC-BY or CC-BY-NC. For more details please see the FathomNet Database Terms of Use.
