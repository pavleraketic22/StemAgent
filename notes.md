# NOTES
**I will be writing my thoughts here while doing this task**

**_28/04/2026_**

My initial idea is to have agent.py file in which Agent as a class is defined and initialized. 

When running main.py, we give task class after which the pipeline is run : explorer -> architect -> (skill_)builder -> stop_condition -> evaluator.

This pipeline generates agent_config.json which basically defines specialized agent.

The biggest issue I have right now is how to make this whole process dynamic. What I mean by that is that I can pre-define some tools, but if I request problem class that needs tools which haven't been pre-defined, how does it fetch and make new tools.

I have a few options:
* Completely fixed number of problem classes and pre-defined tools for each
* Architect chooses tools from pre-defined ones, even when given new problem class
* **Dynamic tool acquisition** Explorer not only looks for the way the problem is solved, but also recommends tools that should be created.
Architect then generates new tool if none of the pre-defined are good enough

I will probably go with the 3rd option:

* tool_generator.py
* tool_library.py
* tool_registry
* tool_specs.json

With these I will make it possible for dynamic tool acquisition

This option is probably the hardest one to implement but the closest one to the task requirements

So idea is:
- give problem class
- explorer searches the web for the way that problem is solved and architecture is built
- architect then uses that info and recommends architecture
- builders build skills, agents and tools
- evaluator evaluates
- and then we have specialized agent !!! 

**_29/04/2026_**

The biggest problem i currently have is to make it possible for a aigent to change its structure while in runtime

I am changing my mind, I'll have tools already defined and ready in a directory called tools.

As for skills it will be .md files that give context to agent how to use tools etc
 
Currently I have a problem seeing how does skills.md help agent, also the way the agent writes each skill.md file is too hardcoded.

What I mean by that is there is always a structure that is followed to the tea and LLM doesn't write anything on its own.

In addition to skills, I have a hard time thinking of ways to evaluate the agent.

I have a few options but none really seem good enough to me> The biggest problem is to make it able to evaluate any problem class.

The options I have rn are:
- LLM judge
- citation score