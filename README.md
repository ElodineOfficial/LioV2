Tl;DR: It's late and I need to stop procrastinating because i can't fix it as quick as i wanted

Goals:
- Agent with multi-faceted personality [done]
- Include user specific memory [done]
- Ability to read all DJ specific channel messages as the LLM [done]
- Respond with gifs [done]
- Vision [not functional at this time]
- DJ capabilities [half done]
- TTS via voice channels [half done]
  
Bugs:
- CV is traaaaaaaaaaaaaaaaaaash and my name is Oscar, two peas in a tin pod bby
- Calls helper personality too often

Upcoming changes:
- CV IF I CAN RUB ENOUGH BRAINCELLS TOGETHER
- Regex patterns/post processing system to fine tune responses from annoying repetition
- Expand emoji library
- DJ capabilities
- Voice
- Replace personality files with final text prompts vs the testing promtps

Current features:
- Announce her presence [when Lio comes online, she posts a greeting in her channel]
- Personality matrix calls one of three different personalities depending on which is most appropriate for your question [a helper, a friend, or the DJ]
- Thread based memory [purges oldest messages at 10k]
- Memory for specific users [gathers user messages every few thousand words and decides which things are important to remember about that person]
- Memory system given a fallback for tricky username/ID combos
- Responds to messages with gifs [currently set to an anime subset, you can theme this to whatever you like by simply changing the anime prefix to anything else]
- Tell users when gifs aren't working with a playful message
- Daily shitposting mechanic
- !bonk command lets you reset recent memory
- Use emojis with each response! [will be moved to a ratio soon]
- Response indicator [now lets you know when Lio is thinking about her response]
