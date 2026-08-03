// vina_mini_chat_welcome.js — compatibility shim
// All welcome logic has moved to elion_mini_chat.js.
// _miniChatWelcome() is kept as an alias so any legacy calls still work.
function _miniChatWelcome() {
    if (typeof _vinaMiniWelcome === 'function') {
        _vinaMiniWelcome();   // delegate to elion_mini_chat.js implementation
    }
}