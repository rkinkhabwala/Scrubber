Scrubber is a **digital black marker**. It reads a piece of text and blacks out anything personal, like names, phone numbers, card numbers and addresses, before that text goes anywhere else.

## The problem it solves

Picture a bank's customer support team. Every day they get thousands of messages like:

> "Hi, I'm Maria Lopez. My card 4111 1111 1111 1111 was charged twice. Call me at 555-201-8890."

The bank wants to use these messages: spot trends, train staff, maybe ask an AI to summarize them. But the messages are full of customers' private details. Legally and ethically, that information can't be shared or sent to an outside AI company.

So someone has to black out the private parts first. Doing that by hand is impossible at that volume, and the existing automatic tools miss a lot. In our test, the most popular free tool still left about **1 in 3** personal details readable.

## What Scrubber does

It turns that message into:

> "Hi, I'm **[NAME_1]**. My card **[CARD_1]** was charged twice. Call me at **[PHONE_1]**."

Now the message is safe to share, store or analyze. The useful part (a double charge happened) is still there, and the private part is gone.

## How we teach it: like training a new employee

**1. Start with a smart but general assistant.** We take a free, open AI model. It's like a new hire who's well read but has never done this job.

**2. Give it practice sheets with an answer key.** We show it thousands of example texts, each with the personal details already marked. That's like handing a new employee a stack of documents where someone already highlighted what needs blacking out, and saying "learn from these."

**3. Add examples from your own workplace.** General examples teach what personal data looks like in general. We add examples that look like the company's real paperwork, such as support tickets, insurance claims and system logs. We also include tricky items that *look* sensitive but aren't, like order numbers or dollar amounts, so it learns not to over-redact.

**4. Give it a final exam.** We test it on documents it has never seen and compare it with the existing tools. The key score is **how much private data slipped through**. Lower is better.

The whole training happens on your own laptop. No customer data is ever uploaded anywhere.

## The safety rule built in

The AI is only allowed to *point* at what's private ("that's a name, that's a card number"). Simple, predictable code does the actual blacking out. So the AI can never accidentally rewrite or change the rest of the document, and if it points at something that isn't really there, the code ignores it.

## Bonus: putting things back

Scrubber remembers which placeholder stood for what, kept privately on the company's side. A bank could send the cleaned message to an outside AI, get back a reply like "Draft a refund note to [NAME_1]," and Scrubber swaps the real name back in at the end. The outside AI never saw it.

**In one line:** Scrubber takes a free AI model, trains it on your own laptop to spot personal information, and turns it into an automatic black marker that keeps private data from leaking, without anything ever leaving your computer.