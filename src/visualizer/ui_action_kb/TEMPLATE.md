# Tool: my_new_tool
# ─────────────────────────────────────────────────────────────────────────────
# Copy this file to ui_action_kb/my_new_tool.md and fill in your tool's actions.
# Restart Flask after saving — the index rebuilds automatically (mtime check).
#
# Rules:
#  - One ## action: section per distinct user intent / button
#  - **btnId:** must exactly match the HTML element id
#  - **triggers:** are natural-language phrases users might say
#    Aim for 8-15 per action; diversity > quantity.
#    Include: direct commands, questions, typos, "then/next/now what" phrases.
#  - **response:** guided text shown in mini-chat. Use {btn} for button name.
#  - The Tool: name must match tool_hint strings passed by routes.py
# ─────────────────────────────────────────────────────────────────────────────

## action: open_visualizer
**btnId:** myOpenBtn
**triggers:**
- open the visualizer
- launch the tool
- show me the interface
- I want to start
- how do I begin
- what is this tool
- open my tool
**response:** Click the flashing **{btn}** button to open the visualizer.

## action: run_analysis
**btnId:** myRunBtn
**triggers:**
- run the analysis
- start processing
- execute
- go ahead
- run it
- now what
- then what
- what do I click next
- next step
**response:** Click **{btn}** to start the analysis. Results will appear below.

## action: load_input
**btnId:** myInputField
**triggers:**
- enter my data
- where do I input
- how do I load data
- set the input
- enter file path
**response:** Enter your input data in the **{btn}** field, then click Run.