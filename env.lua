function get_player_coordinates()
    -- Replace with actual memory addresses for player's X and Y coordinates
    local player_x = memory.read_u16_le(0x02000000)
    local player_y = memory.read_u16_le(0x02000002)
    return player_x, player_y
end

while true do
    local x, y = get_player_coordinates()
    print(string.format("Player Coordinates: X=%d, Y=%d", x, y))
    emu.frameadvance()  -- Advance to the next frame
end