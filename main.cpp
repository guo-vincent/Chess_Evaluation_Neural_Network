#include <iostream>
#include "chess.hpp"
#include <onnxruntime_cxx_api.h>
#include "evaluator.h"
#include "board_wrapper.h"
#include "raylib.h"
#include "rlImGui.h"
#include "imgui.h"

struct GameContext {
    Player whitePlayer = HUMAN;
    Player blackPlayer = HUMAN;
};

enum class GameState {
    MAIN_MENU,
    SELECT_MODE,
    SETTINGS,
    PLAYING,
    EXIT
};

bool ButtonCentered(const char* text, int y, int width, int height, Color colorNormal, Color colorHover) {
    int screenWidth = GetScreenWidth();
    int x = screenWidth / 2 - width / 2;
    Vector2 mouse = GetMousePosition();
    bool hover = mouse.x >= x && mouse.x <= x + width &&
                 mouse.y >= y && mouse.y <= y + height;

    DrawRectangle(x, y, width, height, hover ? colorHover : colorNormal);

    int textWidth = MeasureText(text, 30);
    DrawText(text, x + width / 2 - textWidth / 2, y + height / 2 - 15, 30, BLACK);

    return hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
}

Player ToggleGroupCentered(int y, int buttonWidth, int buttonHeight, int spacing, 
                       const char* option1, const char* option2, 
                       Player activeOption, Color colorNormal, 
                       Color colorActive, Color colorHover) {
    int totalWidth = 2 * buttonWidth + spacing;
    int screenWidth = GetScreenWidth();
    int startX = screenWidth / 2 - totalWidth / 2;

    // Option 1 button
    Rectangle opt1Rect = { (float)startX, (float)y, (float)buttonWidth, (float)buttonHeight };
    bool opt1Hover = CheckCollisionPointRec(GetMousePosition(), opt1Rect);
    bool opt1Clicked = opt1Hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
    Color opt1Color = (activeOption == HUMAN) ? colorActive : (opt1Hover ? colorHover : colorNormal);
    DrawRectangleRec(opt1Rect, opt1Color);
    int textWidth1 = MeasureText(option1, 20);
    DrawText(option1, startX + buttonWidth/2 - textWidth1/2, y + buttonHeight/2 - 10, 20, BLACK);

    // Option 2 button
    Rectangle opt2Rect = { (float)(startX + buttonWidth + spacing), (float)y, (float)buttonWidth, (float)buttonHeight };
    bool opt2Hover = CheckCollisionPointRec(GetMousePosition(), opt2Rect);
    bool opt2Clicked = opt2Hover && IsMouseButtonPressed(MOUSE_LEFT_BUTTON);
    Color opt2Color = (activeOption == BOT) ? colorActive : (opt2Hover ? colorHover : colorNormal);
    DrawRectangleRec(opt2Rect, opt2Color);
    int textWidth2 = MeasureText(option2, 20);
    DrawText(option2, startX + buttonWidth + spacing + buttonWidth/2 - textWidth2/2, y + buttonHeight/2 - 10, 20, BLACK);

    if (opt1Clicked) return HUMAN;
    if (opt2Clicked) return BOT;
    return activeOption;
}

int main() {
    SetConfigFlags(FLAG_FULLSCREEN_MODE);
    InitWindow(0, 0, "Chess UI");
    SetTargetFPS(60);

    rlImGuiSetup(true);

    GameState state = GameState::MAIN_MENU;

    chess::Board board;
    BoardWrapper boardWrapper(board);

    GameContext gameContext;

    while (!WindowShouldClose() && state != GameState::EXIT) {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        if (state == GameState::MAIN_MENU) {
            DrawText("CHESS", GetScreenWidth() / 2 - MeasureText("CHESS", 60) / 2, 150, 60, BLACK);

            if (ButtonCentered("Play", 300, 200, 60, LIGHTGRAY, GRAY)) {
                boardWrapper.init();
                state = GameState::SELECT_MODE;
            }
            if (ButtonCentered("Exit", 400, 200, 60, LIGHTGRAY, GRAY)) {
                state = GameState::EXIT;
            }

        } else if (state == GameState::SELECT_MODE) {
            const char* title = "SELECT OPTIONS";
            int titleWidth = MeasureText(title, 60);
            DrawText(title, GetScreenWidth() / 2 - titleWidth / 2, 150, 60, BLACK);

            int currentY = 220;

            // White player selection
            const char* whiteLabel = "White:";
            DrawText(whiteLabel, GetScreenWidth() / 2 - MeasureText(whiteLabel, 30) / 2, currentY, 30, BLACK);
            currentY += 40;
            gameContext.whitePlayer = ToggleGroupCentered(currentY, 100, 40, 10, "Human", "Bot",
                                                          gameContext.whitePlayer, LIGHTGRAY, DARKGRAY, GRAY);
            currentY += 60;

            const char* blackLabel = "Black:";
            DrawText(blackLabel, GetScreenWidth() / 2 - MeasureText(blackLabel, 30) / 2, currentY, 30, BLACK);
            currentY += 40;
            gameContext.blackPlayer = ToggleGroupCentered(currentY, 100, 40, 10, "Human", "Bot",
                                                          gameContext.blackPlayer, LIGHTGRAY, DARKGRAY, GRAY);
            currentY += 60;

            // Play button
            if (ButtonCentered("Play", currentY, 200, 60, LIGHTGRAY, GRAY)) {
                boardWrapper.init(gameContext.whitePlayer, gameContext.blackPlayer);
                state = GameState::PLAYING;
            }
        } else if (state == GameState::PLAYING) {
            boardWrapper.UpdateClicks();
            boardWrapper.DrawBoard();

            rlImGuiBegin();
            if (ButtonCentered("Back to Menu", 20, 200, 40, LIGHTGRAY, GRAY)) {
                boardWrapper.stopBotThread();
                state = GameState::MAIN_MENU;
            }
            if (ButtonCentered("Reset", 70, 200, 40, LIGHTGRAY, GRAY)) {
                boardWrapper.stopBotThread();
                ImGui::OpenPopup("Reset Game?");
            }

            if (ImGui::BeginPopupModal("Reset Game?", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
                ImGui::Text("Do you want to:\n\n"
                            "- Continue where you left off\n"
                            "- Restart from the beginning?");
                ImGui::Separator();

                if (ImGui::Button("Continue", ImVec2(120, 0))) {
                    ImGui::CloseCurrentPopup(); 
                }
                ImGui::SameLine();
                if (ImGui::Button("Restart", ImVec2(120, 0))) {
                    boardWrapper.reset();
                    ImGui::CloseCurrentPopup();
                }

                ImGui::EndPopup();
            }
            rlImGuiEnd();
        }

        EndDrawing();
    }

    rlImGuiShutdown();
    CloseWindow();
    return 0;
}