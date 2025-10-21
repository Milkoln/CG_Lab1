#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

#define M_PI 3.1415926535897932384626433832795

namespace {

constexpr float camera_fov = 70.0f;
constexpr float camera_near_plane = 0.01f;
constexpr float camera_far_plane = 100.0f;

struct Matrix {
	float m[4][4];
};

struct Vector {
	float x, y, z;
};

struct Vertex {
	Vector position;
	//Vector color;
	// NOTE: You can add more attributes
};

// NOTE: These variable will be available to shaders through push constant uniform
struct ShaderConstants {
	Matrix projection;
	Matrix transform;
	Vector color;
};

struct VulkanBuffer {
	VkBuffer buffer;
	VkDeviceMemory memory;
};

VkShaderModule vertex_shader_module;
VkShaderModule fragment_shader_module;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;

// NOTE: Declare buffers and other variables here
VulkanBuffer cube_vertex_buffer;
VulkanBuffer cube_index_buffer;

VulkanBuffer vertex_sphere_buffer;
VulkanBuffer index_sphere_buffer;

Vector cube_position = {0.0f, 0.0f, 0.0f};
float cube_rotation;
Vector cube_color = {0.5f, 1.0f, 0.7f };
bool cube_spin = false;

Vector sphere_position = { 0.0f, 0.0f, 5.0f };
float sphere_rotation;
Vector sphere_color = { 1.0f, 0.7f, 0.4f };
bool sphere_spin = true;

// Sphere variables
const uint32_t n = 30; // Number of longituded and latitudes
const uint32_t num_of_triangles = (2 * n + 2 * n * (n - 1));

Vertex vertices_sphere[n * n + 2];
uint32_t indices_sphere[3 * num_of_triangles];

float phi, theta; // Spherical coordinates
float R = 1.2f;
Vertex sphere_vert;

Matrix identity() {
	Matrix result{};

	result.m[0][0] = 1.0f;
	result.m[1][1] = 1.0f;
	result.m[2][2] = 1.0f;
	result.m[3][3] = 1.0f;
	
	return result;
}

Matrix projection(float fov, float aspect_ratio, float near, float far) {
	Matrix result{};

	const float radians = fov * M_PI / 180.0f;
	const float cot = 1.0f / tanf(radians / 2.0f);

	result.m[0][0] = cot / aspect_ratio;
	result.m[1][1] = cot;
	result.m[2][3] = 1.0f;

	result.m[2][2] = far / (far - near);
	result.m[3][2] = (-near * far) / (far - near);

	return result;
}

Matrix translation(Vector vector) {
	Matrix result = identity();

	result.m[3][0] = vector.x;
	result.m[3][1] = vector.y;
	result.m[3][2] = vector.z;

	return result;
}

Matrix rotation(Vector axis, float angle) {
	Matrix result{};

	float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

	axis.x /= length;
	axis.y /= length;
	axis.z /= length;

	float sina = sinf(angle);
	float cosa = cosf(angle);
	float cosv = 1.0f - cosa;

	result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
	result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
	result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

	result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
	result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
	result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

	result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
	result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
	result.m[2][2] = (axis.z * axis.z * cosv) + cosa;

	result.m[3][3] = 1.0f;

	return result;
}

Matrix multiply(const Matrix& a, const Matrix& b) {
	Matrix result{};

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			for (int k = 0; k < 4; k++) {
				result.m[j][i] += a.m[j][k] * b.m[k][i];
			}
		}
	}

	return result;
}

Matrix camera_position = translation({ 0.0f, 2.0f, 10.0f });

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;
	
	VulkanBuffer result{};

	{
		// NOTE: We create a buffer of specific usage with specified size
		VkBufferCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size,
			.usage = usage,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};

		if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan buffer\n";
			return {};
		}
	}

	// NOTE: Creating a buffer does not allocate memory,
	//       only a buffer **object** was created.
	//       So, we allocate memory for the buffer

	{
		// NOTE: Ask buffer about its memory requirements
		VkMemoryRequirements requirements;
		vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

		// NOTE: Ask GPU about types of memory it supports
		VkPhysicalDeviceMemoryProperties properties;
		vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

		// NOTE: We want type of memory which is visible to both CPU and GPU
		// NOTE: HOST is CPU, DEVICE is GPU; we are interested in "CPU" visible memory
		// NOTE: COHERENT means that CPU cache will be invalidated upon mapping memory region
		const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

		// NOTE: Linear search through types of memory until
		//       one type matches the requirements, thats the index of memory type
		uint32_t index = UINT_MAX;
		for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
			const VkMemoryType& type = properties.memoryTypes[i];

			if ((requirements.memoryTypeBits & (1 << i)) &&
			    (type.propertyFlags & flags) == flags) {
				index = i;
				break;
			}
		}

		if (index == UINT_MAX) {
			std::cerr << "Failed to find required memory type to allocate Vulkan buffer\n";
			return {};
		}

		// NOTE: Allocate required memory amount in appropriate memory type
		VkMemoryAllocateInfo info{
			.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
			.allocationSize = requirements.size,
			.memoryTypeIndex = index,
		};

		if (vkAllocateMemory(device, &info, nullptr, &result.memory) != VK_SUCCESS) {
			std::cerr << "Failed to allocate Vulkan buffer memory\n";
			return {};
		}

		// NOTE: Link allocated memory with a buffer
		if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) {
			std::cerr << "Failed to bind Vulkan  buffer memory\n";
			return {};
		}

		// NOTE: Get pointer to allocated memory
		void* device_data;
		vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);

		memcpy(device_data, data, size);

		vkUnmapMemory(device, result.memory);
	}

	return result;
}

void destroyBuffer(const VulkanBuffer& buffer) {
	VkDevice& device = veekay::app.vk_device;
	vkFreeMemory(device, buffer.memory, nullptr);
	vkDestroyBuffer(device, buffer.buffer, nullptr);
}

// Modified remain (костыль)
uint32_t r(const uint32_t x, const uint32_t n) {
	if (x == n * n) { 
		return x; 
	}
	else { 
		return (x % (n * n)); 
	}
	return 0;
}

void initialize() {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			// NOTE: If you want more attributes per vertex, declare them here
			/*{
				.location = 1, // NOTE: Second attribute
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, color),
			}, */
		};

		// NOTE: Bring 
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		// NOTE: Declare constant memory region visible to vertex and fragment shaders
		VkPushConstantRange push_constants{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
			              VK_SHADER_STAGE_FRAGMENT_BIT,
			.size = sizeof(ShaderConstants),
		};

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_constants,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	// TODO: You define cube vertices and create buffers here
	// TODO: Index buffer has to be created here too
	// NOTE: Look for createBuffer function

	// (v0)------(v1)
	//  |  \       |
	//  |   `--,   |
	//  |       \  |
	// (v3)------(v2)
	
	Vertex cube_vertices[] = {
		{{-1.0f, -1.0f, 0.0f}},
		{{1.0f, -1.0f, 0.0f}},
		{{1.0f, 1.0f, 0.0f}},
		{{-1.0f, 1.0f, 0.0f}},
		{{-1.0f, -1.0f, 2.0f}},
		{{1.0f, -1.0f, 2.0f}},
		{{1.0f, 1.0f, 2.0f}},
		{{-1.0f, 1.0f, 2.0f}},
	};

	uint32_t cube_indices[] = 
	{ 0, 1, 2, 2, 3, 0, 
		4, 5, 1, 1, 0, 4, 
		4, 0, 3, 3, 7, 4, 
		1, 5, 6, 6, 2, 1, 
		3, 2, 6, 6, 7, 3, 
		7, 6, 5, 5, 4, 7, 
	};

	cube_vertex_buffer = createBuffer(sizeof(cube_vertices), cube_vertices,
	                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	cube_index_buffer = createBuffer(sizeof(cube_indices), cube_indices,
	                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

	// SPHERE CREATION
	// Adding vertices to vertices_sphere;
	for (uint32_t phi_i = 0; phi_i <= n - 1; phi_i++) { // Defines the "layer lvl"
		phi = phi_i * 2 * M_PI / n;
		for (uint32_t theta_i = 1; theta_i <= n; theta_i++) {
			theta = (-M_PI / 2) + theta_i * M_PI / (n + 1);
			sphere_vert.position = { R * cos(theta) * cos(phi), R * cos(theta) * sin(phi), R * sin(theta) };
			vertices_sphere[theta_i + n * phi_i] = sphere_vert;
		}
	}
	// Add South and North pole
	Vertex South_pole = { { 0, 0, -R } };
	Vertex North_pole = { { 0, 0, R } };
	vertices_sphere[0] = South_pole;
	vertices_sphere[n * n + 1] = North_pole;

	// Adding triangles to indices_sphere
	uint32_t trangles_vert_added = 0;
	// Pt1: South Pole connected to the first (lowerest) layer. So we add n trangles and their n insides
	for (uint32_t i = 0; i <= n - 1; i++) {
		indices_sphere[trangles_vert_added] = 1 + n * i;
		indices_sphere[trangles_vert_added + 1] = r(1 + n * i + n, n);
		indices_sphere[trangles_vert_added + 2] = 0;
		trangles_vert_added += 3;
	}

	// Pt2: k layer connected to the (k-1) layer. So at each layer we add 2n triangles. 
	// And we have n - 1 layer. So 2n(n-1) trangle 
	for (uint32_t k = 2; k <= n; k++) {		
		for (uint32_t j = 1; j <= n; j++) {
			indices_sphere[trangles_vert_added] = k + n * (j - 1);
			indices_sphere[trangles_vert_added + 1] = r(k + n * j, n);
			indices_sphere[trangles_vert_added + 2] = r(k + n * j, n) - 1;
			indices_sphere[trangles_vert_added + 3] = r(k + n * j, n) - 1;
			indices_sphere[trangles_vert_added + 4] = k - n + n * j - 1;
			indices_sphere[trangles_vert_added + 5] = k + n * j - n;
			trangles_vert_added += 6;
		}
	}

	// Pt3: North Pole connected to the n_th layer. So we add n trangles and their n insides. Same as the South pole
	for (uint32_t i = 1; i <= n; i++) {
		indices_sphere[trangles_vert_added] = n * n + 1;
		indices_sphere[trangles_vert_added + 1] = r(n * (i + 1), n);
		indices_sphere[trangles_vert_added + 2] = n * i;
		trangles_vert_added += 3;
	}

	vertex_sphere_buffer = createBuffer(sizeof(vertices_sphere), vertices_sphere,
		VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

	index_sphere_buffer = createBuffer(sizeof(indices_sphere), indices_sphere,
		VK_BUFFER_USAGE_INDEX_BUFFER_BIT); 
}

void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	// NOTE: Destroy resources here, do not cause leaks in your program!
	destroyBuffer(cube_index_buffer);
	destroyBuffer(cube_vertex_buffer);

	destroyBuffer(index_sphere_buffer);
	destroyBuffer(vertex_sphere_buffer);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	ImGui::Begin("Controls:");
	/*ImGui::InputFloat3("Translation", reinterpret_cast<float*>(&cube_position));
	ImGui::SliderFloat("Rotation", &cube_rotation, 0.0f, 2.0f * M_PI); */
	ImGui::InputFloat3("Translation", reinterpret_cast<float*>(&sphere_position));
	ImGui::SliderFloat("Rotation", &sphere_rotation, 0.0f, 2.0f * M_PI);
	ImGui::Checkbox("Spin?", &cube_spin);
	// TODO: Your GUI stuff here
	ImGui::End();

	// NOTE: Animation code and other runtime variable updates go here
	if (cube_spin) {
		cube_rotation = float(time);
	}

	cube_rotation = fmodf(cube_rotation, 2.0f * M_PI);

	if (sphere_spin) {
		sphere_rotation = float(time);
	}

	sphere_rotation = fmodf(sphere_rotation, 2.0f * M_PI);
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	// TODO: Vulkan rendering code here
	// NOTE: ShaderConstant updates, vkCmdXXX expected to be here
	{
		// NOTE: Use our new shiny graphics pipeline
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		// NOTE: Use our quad vertex buffer
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &cube_vertex_buffer.buffer, &offset);

		// NOTE: Use our quad index buffer
		vkCmdBindIndexBuffer(cmd, cube_index_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);

		// NOTE: Variables like cube_XXX were declared globally
		ShaderConstants cube_constants{
			.projection = projection(
				camera_fov,
				float(veekay::app.window_width) / float(veekay::app.window_height),
				camera_near_plane, camera_far_plane),

			.transform = multiply(multiply(rotation({0.0f, 1.0f, 0.0f}, cube_rotation),
			                      translation(cube_position)), camera_position),

			//.color = cube_color,
			.color = {0.0f, 1.0f, 0.1f},
		};

		// NOTE: Update constant memory with new shader constants
		vkCmdPushConstants(cmd, pipeline_layout,
		                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		                   0, sizeof(ShaderConstants), &cube_constants);

		// NOTE: Draw 3N indices (3 vertices * N triangles), 1 group, no offsets
		vkCmdDrawIndexed(cmd, 36, 1, 0, 0, 0);

		// SPHERE
		vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_sphere_buffer.buffer, &offset);
		vkCmdBindIndexBuffer(cmd, index_sphere_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);

		// NOTE: Variables like cube_XXX were declared globally
		ShaderConstants sphere_constants{
			.projection = projection(
				camera_fov,
				float(veekay::app.window_width) / float(veekay::app.window_height),
				camera_near_plane, camera_far_plane),

			.transform = multiply(multiply(translation(sphere_position), 
				rotation({0.0f, 1.0f, 0.0f}, sphere_rotation)), camera_position),

				.color = {1.0f, 0.0f, 0.5f},
		};

		// NOTE: Update constant memory with new shader constants
		vkCmdPushConstants(cmd, pipeline_layout,
			VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(ShaderConstants), &sphere_constants);

		// NOTE: Draw 3N indices (3 vertices * N triangles), 1 group, no offsets
		vkCmdDrawIndexed(cmd, 3 * num_of_triangles, 1, 0, 0, 0); 
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
