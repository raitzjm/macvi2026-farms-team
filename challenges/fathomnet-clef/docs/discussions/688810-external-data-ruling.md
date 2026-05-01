# Discussion #688810 — Question Regarding Rule 6: External Data and Tools

**URL:** https://www.kaggle.com/competitions/fathomnet-2026/discussion/688810  
**Posted by:** Rob Brock  
**Date:** ~19 days before 2026-04-27 (approx. 2026-04-08)  
**Status:** Resolved — official organizer ruling by Laura Chrobak (Competition Host)

---

## Original Post (Rob Brock)

Hey there,

I'm really enjoying the problem so far! I have some questions about how both the training data and external can be used.

Rule 6 states:

> a. The general rule is that participants should only use the provided training and validation data for training models to classify the test images. We do not want participants crawling the web in search of additional data. Pretrained models may be used to construct the algorithms from publicly available academic datasets (e.g., ImageNet, iNaturalist). Please specify (openly in Discussion section) any and all external data and/or models used for training before uploading results.

Obviously this would prohibit training a model with images and labels of species that do not belong in the training set of data. The rule says to only use the provided training and validation data for training models to classify the test images. Does classification also include object detection (i.e. a model that detects "object" and not "species/class")?

I have a handful of ideas but before moving forward with them I wanted to make sure of which of the below were considered "off-limits" for the competition:

- Is use of the FathomNet Megalodon model allowed?
- Is use of other object detection models that focus on marine life (i.e. Megafishdetector) allowed, provided they are trained on publicly available datasets?
- Is use of other FathomNet Database images for object detection allowed if the species/classes are omitted from the bounding boxes and labels?
- Are we allowed to label existing training/validation data to some extent ourselves (i.e. object/not-object)?

I just want to be sure I'm not over-reaching or bending/breaking any rules as we move forward and continue to explore and develop a solution.

Thanks for the guidance!

Rob

---

## Reply 1 — Laura Chrobak (Competition Host, ~19 days ago)

Hi @robbrock,

Great questions - and really glad to hear you're enjoying the problem so far!

You're interpreting Rule 6 correctly in spirit: the goal is to ensure a fair comparison by having all participants train their task-specific models using only the provided training and validation data, rather than expanding the dataset with additional labeled examples from elsewhere.

To clarify your specific questions:

**Object detection vs. classification:** For the purposes of the competition, "classification" should be interpreted broadly to include any task-specific training for this problem - including object detection.

**Use of the FathomNet Megalodon model:** Using pretrained models like this is allowed as a starting point (similar to ImageNet-style pretraining). Any use should be clearly documented.

**Other pretrained marine-life detection models (e.g., Megafishdetector):** These are allowed for initialization or feature extraction if they are publicly available.

**Using additional FathomNet images (without labels):** This would still be considered augmenting the dataset and is not allowed, even if labels are omitted or simplified.

**Relabeling provided training/validation data:** Yes — this is allowed. You're welcome to derive additional annotations (e.g., object vs. not-object) from the provided dataset as long as you're only using the competition data itself.

In short:

- ✅ Pretrained models: allowed (with disclosure)
- ❌ External labeled data (even indirectly): not allowed
- ✅ Transforming/relabeling provided data: allowed

And as you noted, transparency is key - please document any external models or resources used in your submission.

Happy to clarify further if anything is still ambiguous - and excited to see what you build!

Cheers,  
Laura

---

## Reply 2 — Rob Brock (~19 days ago)

Sounds good, and thank you for the clarifications! There was another question posted here but it seems to be gone now. They'd pointed out that the Guidelines on the Overview page states:

> Participants should not use other image sets to train their models outside of the FathomNet-CLEF Kaggle competition data and the FathomNet Database website, however they can leverage other types of outside data sources for other aspects of the competition. Pre-trained models can be used for initialization of training. Teams should plan on specifying additional data sources and/or pre-trained models used when uploading results.

So there seems to be a misalignment between what the Overview's Guidelines say and what is on the Rules page, since the guidelines section states to not use data "outside of the FathomNet-CLEF Kaggle competition data and the FathomNet Database website."

---

## Reply 3 — Laura Chrobak (Competition Host, ~18 days ago)

Hi @robbrock,

Great catch - thanks for flagging this.

You're right that there's ambiguity between the Overview and Rules language. Given this, and the advanced stage of the competition, we want to avoid introducing new restrictions or reinterpretations that could impact teams already developing their approaches.

At this point, the key expectation is transparency: if pretrained models or any external resources are used, they should be clearly disclosed in your submission.

We'll work on clarifying the language for future iterations, but for now, please make sure to document anything external you use.

Thanks again for calling this out!

Best,  
Laura

---

## Summary & Ruling for Agents

| Resource | Status | Source |
|---|---|---|
| Pretrained models (ImageNet, iNaturalist, COCO, etc.) | ✅ Allowed — must disclose | Laura reply 1 |
| FathomNet Megalodon model (pretrained) | ✅ Allowed — must disclose | Laura reply 1 |
| Megafishdetector and similar public marine models | ✅ Allowed for initialization/feature extraction | Laura reply 1 |
| Relabeling / re-annotating competition data (e.g. object/not-object) | ✅ Allowed | Laura reply 1 |
| Additional FathomNet Database images (even without labels) | ❌ Not allowed | Laura reply 1 |
| Other external image datasets | ❌ Not allowed | Rule 6a |
| Web crawling for additional data | ❌ Not allowed | Rule 6a |

**Key organizer stance (Laura's final message):** Due to ambiguity between the Guidelines (which mention FathomNet Database as allowed) and the Rules page (competition data only), organizers chose not to introduce new restrictions mid-competition. **Transparency/disclosure is the primary enforcement mechanism.** Document everything external in the Kaggle Discussion tab before submitting.
