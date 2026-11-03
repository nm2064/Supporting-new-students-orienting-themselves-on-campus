# UniBot frontend design

The interface helps new students find a place, an event or the right support, especially on a phone. Keep existing chat, language, citation and routing behavior.

Use UniBot blue for actions, pale neutral page backgrounds, white or dark navy reading surfaces, and real university photography. Images are campus context, not claims about a particular event. Orkney's photograph depicts Skara Brae and is labelled accordingly. Image origins and credits are in images/campus/sources.json.

Home has a photo-led campus welcome and three discovery cards. Events use date blocks, venue information and explicit source/map actions. Services use category icons, short previews and expandable descriptions. Chat has optional question starters which fill the composer without sending. On phones, map controls stack above the full-width map.

Use one-column content on small screens, 44px controls, visible keyboard focus, RTL layout support and existing dark mode. Preserve API loading, empty and error states. Browser tests cover 360px, 390px, 768px and 1440px widths; they do not replace device testing or accessibility review.
